// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: 2019-2024 Second State INC

//===-- wasmedge/runtime/stackmgr.h - Stack Manager definition ------------===//
//
// Part of the WasmEdge Project.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of Stack Manager.
///
//===----------------------------------------------------------------------===//
#pragma once

#include "ast/instruction.h"
#include "common/types.h"
#include "gc/allocator.h"
#include "runtime/instance/module.h"

#include <cstring>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// The SIMD element types stored on the value stack and popped through tuples.
#define WASMEDGE_FOR_EACH_SIMD_TYPE(X)                                         \
  X(int64x2_t)                                                                 \
  X(uint64x2_t)                                                                \
  X(int32x4_t)                                                                 \
  X(uint32x4_t)                                                                \
  X(int16x8_t)                                                                 \
  X(uint16x8_t) X(int8x16_t) X(uint8x16_t) X(doublex2_t) X(floatx4_t)

// clang-cl makes alignof(std::tuple<uint64x2_t>) 8 instead of 16, so an aligned
// SIMD load on a tuple slot can fault. MS-STL stores each tuple element in a
// std::_Tuple_val<T>, so over-aligning that wrapper restores 16-byte alignment
// and lets StackManager use plain std::tuple. Only needed for clang-cl (native
// Clang already aligns; MSVC uses std::array).
// Ref: https://github.com/llvm/llvm-project/issues/134694
#if defined(_MSC_VER) && defined(__clang__)
namespace std {
#define WASMEDGE_X(T)                                                          \
  template <> struct alignas(WasmEdge::T) _Tuple_val<WasmEdge::T> {            \
    constexpr _Tuple_val() : _Val() {}                                         \
    template <typename OtherT>                                                 \
    constexpr _Tuple_val(OtherT &&Arg) : _Val(std::forward<OtherT>(Arg)) {}    \
    WasmEdge::T _Val;                                                          \
  };
WASMEDGE_FOR_EACH_SIMD_TYPE(WASMEDGE_X)
#undef WASMEDGE_X
} // namespace std
#endif

// A std::tuple holding a SIMD element must keep that element's alignment (see
// the issue above). Outside the clang-cl guard so any future regression fails
// here at compile time instead of crashing at runtime.
#define WASMEDGE_X(T)                                                          \
  static_assert(alignof(std::tuple<WasmEdge::T>) == alignof(WasmEdge::T),      \
                "tuple<" #T "> alignment mismatch");
WASMEDGE_FOR_EACH_SIMD_TYPE(WASMEDGE_X)
#undef WASMEDGE_X
#undef WASMEDGE_FOR_EACH_SIMD_TYPE

namespace WasmEdge {
namespace Runtime {

class StackManager {
public:
  using Value = ValVariant;

  struct Handler {
    Handler(AST::InstrView::iterator TryIt, uint32_t V,
            Span<const AST::Instruction::CatchDescriptor> C)
        : Try(TryIt), VPos(V), CatchClause(C) {}
    AST::InstrView::iterator Try;
    uint32_t VPos;
    Span<const AST::Instruction::CatchDescriptor> CatchClause;
  };

  struct Frame {
    Frame() = delete;
    Frame(const Instance::ModuleInstance *Mod, AST::InstrView::iterator FromIt,
          uint32_t L, uint32_t A, uint32_t V) noexcept
        : Module(Mod), From(FromIt), Arity(A), VPos(V), Locals(L) {}
    const Instance::ModuleInstance *Module;
    AST::InstrView::iterator From;
    uint32_t Arity;
    uint32_t VPos;
    uint32_t Locals;
    std::vector<Handler> HandlerStack;
  };

  /// Stack manager provides the stack control for Wasm execution with VALIDATED
  /// modules. All operations of instructions passed validation, therefore no
  /// unexpect operations will occur.
  explicit StackManager(GC::Allocator &A) noexcept : Allocator(A) {
    ValueStack.reserve(2048U);
    FrameStack.reserve(16U);
    Allocator.addStack(ValueStack);
  }
  ~StackManager() noexcept { Allocator.removeStack(ValueStack); }

  // ValueStack is registered with the GC by address (addStack); a copy or move
  // would leave the new object's stack unregistered (its refs invisible to the
  // collector -> premature reclaim) and make the destructor's removeStack
  // target an address the allocator never stored. Non-copyable and non-movable.
  StackManager(const StackManager &) = delete;
  StackManager(StackManager &&) = delete;
  StackManager &operator=(const StackManager &) = delete;
  StackManager &operator=(StackManager &&) = delete;

  /// Getter for stack size.
  size_t size() const noexcept { return ValueStack.size(); }

  /// Push a new value entry to the stack. Marked noexcept by policy: an
  /// allocation failure on the growth path aborts (std::terminate) rather than
  /// unwinding, since the GC-registered value stack cannot be left in a
  /// partially-grown state mid-collection.
  template <typename T> void push(T &&Val) noexcept {
    // Reallocating past capacity frees the buffer the collector's root scan may
    // be iterating; serialize against it on that rare growth path only (the
    // collector reads this GC-registered vector under the allocator's stack
    // lock). The common in-capacity push stays lock-free. See lockStackRoots.
    if (unlikely(ValueStack.size() == ValueStack.capacity())) {
      auto Lock = Allocator.lockStackRoots();
      ValueStack.push_back(std::forward<T>(Val));
    } else {
      ValueStack.push_back(std::forward<T>(Val));
    }
  }

  /// Pop and return the top entry.
  template <typename T> T pop() noexcept {
    assuming(!ValueStack.empty());
    Value V = std::move(ValueStack.back());
    ValueStack.pop_back();
    return get<T>(V);
  }

  /// Pop the top entries into a tuple; the first template argument is the
  /// topmost entry. Brace initialization evaluates left-to-right, so the
  /// pop<>() calls run in template-parameter order (a well-defined pop order).
  template <typename... ArgsT> std::tuple<ArgsT...> pops() noexcept {
    return std::tuple<ArgsT...>{pop<ArgsT>()...};
  }

  /// Getter of top entry of stack.
  template <typename T> T peekTop() const noexcept {
    assuming(!ValueStack.empty());
    return get<T>(ValueStack.back());
  }

  /// Getter of top N-th value entry of stack.
  template <typename T> T peekTopN(uint32_t Offset) const noexcept {
    assuming(0 < Offset && Offset <= ValueStack.size());
    return get<T>(ValueStack[ValueStack.size() - Offset]);
  }

  /// Pop the I-th operand, or peek (leave on the stack) the last one.
  template <std::size_t I, std::size_t Last, typename Ty>
  Ty popOrPeek() noexcept {
    if constexpr (I == Last) {
      return peekTop<Ty>();
    } else {
      return pop<Ty>();
    }
  }
  template <typename... ArgsT, std::size_t... Is>
  std::tuple<ArgsT...> popsPeekTopImpl(std::index_sequence<Is...>) noexcept {
    return std::tuple<ArgsT...>{
        popOrPeek<Is, sizeof...(ArgsT) - 1, ArgsT>()...};
  }
  /// Pop the top entries into a tuple but leave the last one on the stack
  /// (peeked, not popped). The first template argument is the topmost entry;
  /// the final argument is the peeked survivor -- used to keep a ref-typed
  /// operand rooted on the value stack while the shallower operands are popped.
  template <typename T, typename... ArgsT>
  std::tuple<T, ArgsT...> popsPeekTop() noexcept {
    return popsPeekTopImpl<T, ArgsT...>(std::index_sequence_for<T, ArgsT...>{});
  }

  /// Setter of top value entry of stack.
  template <typename T> void emplaceTop(T &&Val) noexcept {
    assuming(!ValueStack.empty());
    emplace(ValueStack.back(), std::forward<T>(Val));
  }

  /// Setter of top N-th value entry of stack.
  template <typename T> void emplaceTopN(uint32_t Offset, T &&Val) noexcept {
    assuming(0 < Offset && Offset <= ValueStack.size());
    emplace(ValueStack[ValueStack.size() - Offset], std::forward<T>(Val));
  }

  /// Push a span of values to the stack.
  void pushSpan(Span<const Value> ValSpan) noexcept {
    // See push(): hold the allocator's stack lock only when the insert grows
    // past capacity and reallocates the buffer the collector may be scanning.
    if (unlikely(ValueStack.size() + ValSpan.size() > ValueStack.capacity())) {
      auto Lock = Allocator.lockStackRoots();
      ValueStack.insert(ValueStack.end(), ValSpan.begin(), ValSpan.end());
    } else {
      ValueStack.insert(ValueStack.end(), ValSpan.begin(), ValSpan.end());
    }
  }

  /// Pop the top ValSpan.size() entries into ValSpan. ValSpan[0] receives the
  /// deepest (earliest-pushed) of the popped entries, ValSpan.back() the top.
  void popSpan(Span<Value> ValSpan) noexcept {
    assuming(ValSpan.size() <= ValueStack.size());
    const auto VSBegin =
        ValueStack.end() - static_cast<ptrdiff_t>(ValSpan.size());
    std::move(VSBegin, ValueStack.end(), ValSpan.begin());
    ValueStack.erase(VSBegin, ValueStack.end());
  }

  /// Unsafe view of the top N value entries while they remain on the stack.
  /// Used at the host- and compiled-function boundary so the arguments stay on
  /// the (GC-rooted) value stack for the duration of the call: moving them into
  /// a detached buffer would leave GC-managed refs unrooted while the callee
  /// runs, and a concurrent collection could then reclaim them.
  /// The returned span aliases the value-stack buffer and dangles after any
  /// operation that grows the stack (push/pushSpan reallocation); do not hold
  /// it across such operations.
  Span<Value> getTopSpan(uint32_t N) noexcept {
    assuming(N <= ValueStack.size());
    return Span<Value>(ValueStack.end() - static_cast<ptrdiff_t>(N), N);
  }

  /// Push a new frame entry to the stack.
  void pushFrame(const Instance::ModuleInstance *Module,
                 AST::InstrView::iterator From, uint32_t Locals = 0,
                 uint32_t Arity = 0, bool IsTailCall = false) noexcept {
    if (!IsTailCall) {
      FrameStack.emplace_back(Module, From, Locals, Arity,
                              static_cast<uint32_t>(ValueStack.size()));
    } else {
      assuming(!FrameStack.empty());
      assuming(FrameStack.back().VPos >= FrameStack.back().Locals);
      assuming(FrameStack.back().VPos - FrameStack.back().Locals <=
               ValueStack.size() - Locals);
      ValueStack.erase(ValueStack.begin() +
                           (FrameStack.back().VPos - FrameStack.back().Locals),
                       ValueStack.end() - Locals);
      FrameStack.back().Module = Module;
      FrameStack.back().Locals = Locals;
      FrameStack.back().Arity = Arity;
      FrameStack.back().VPos = static_cast<uint32_t>(ValueStack.size());
      FrameStack.back().HandlerStack.clear();
    }
  }

  /// Pop top frame.
  AST::InstrView::iterator popFrame() noexcept {
    assuming(!FrameStack.empty());
    assuming(FrameStack.back().VPos >= FrameStack.back().Locals);
    assuming(FrameStack.back().VPos - FrameStack.back().Locals <=
             ValueStack.size() - FrameStack.back().Arity);
    ValueStack.erase(ValueStack.begin() +
                         (FrameStack.back().VPos - FrameStack.back().Locals),
                     ValueStack.end() - FrameStack.back().Arity);
    auto From = FrameStack.back().From;
    FrameStack.pop_back();
    return From;
  }

  // Get all frames
  Span<const Frame> getFramesSpan() const { return FrameStack; }

  /// Push handler for try-catch block.
  void
  pushHandler(AST::InstrView::iterator TryIt, uint32_t BlockParamNum,
              Span<const AST::Instruction::CatchDescriptor> Catch) noexcept {
    assuming(!FrameStack.empty());
    FrameStack.back().HandlerStack.emplace_back(
        TryIt, static_cast<uint32_t>(ValueStack.size()) - BlockParamNum, Catch);
  }

  /// Pop the top handler on the stack.
  std::optional<Handler> popTopHandler(uint32_t AssocValSize) noexcept {
    while (!FrameStack.empty()) {
      auto &Frame = FrameStack.back();
      if (!Frame.HandlerStack.empty()) {
        auto TopHandler = std::move(Frame.HandlerStack.back());
        Frame.HandlerStack.pop_back();
        assuming(TopHandler.VPos <= ValueStack.size() - AssocValSize);
        ValueStack.erase(ValueStack.begin() + TopHandler.VPos,
                         ValueStack.end() - AssocValSize);
        return TopHandler;
      }
      FrameStack.pop_back();
    }
    return std::nullopt;
  }

  /// Remove inactive handler.
  void removeInactiveHandler(AST::InstrView::iterator PC) noexcept {
    assuming(!FrameStack.empty());
    // First pop the inactive handlers. Br instructions may cause the handlers
    // in current frame becomes inactive.
    auto &HandlerStack = FrameStack.back().HandlerStack;
    while (!HandlerStack.empty()) {
      auto &Handler = HandlerStack.back();
      if (PC < Handler.Try ||
          PC > Handler.Try + Handler.Try->getTryCatch().JumpEnd) {
        HandlerStack.pop_back();
      } else {
        break;
      }
    }
  }

  /// Erase value stack.
  void eraseValueStack(uint32_t EraseBegin, uint32_t EraseEnd) noexcept {
    assuming(EraseEnd <= EraseBegin && EraseBegin <= ValueStack.size());
    ValueStack.erase(ValueStack.end() - EraseBegin,
                     ValueStack.end() - EraseEnd);
  }

  // Get all Value
  Span<const Value> getValueSpan() const { return ValueStack; }

  /// Leave top label.
  AST::InstrView::iterator
  maybePopFrameOrHandler(AST::InstrView::iterator PC) noexcept {
    if (FrameStack.size() > 1 && PC->isExprLast()) {
      // Noted that there's always a base frame in stack.
      return popFrame();
    }
    if (PC->isTryBlockLast()) {
      FrameStack.back().HandlerStack.pop_back();
    }
    return PC;
  }

  /// Getter of module address.
  const Instance::ModuleInstance *getModule() const noexcept {
    if (unlikely(FrameStack.empty())) {
      return nullptr;
    }
    return FrameStack.back().Module;
  }

  /// Reset stack.
  void reset() noexcept {
    ValueStack.clear();
    FrameStack.clear();
  }

private:
  /// \name Data of the stack manager.
  /// @{
  std::vector<Value> ValueStack;
  std::vector<Frame> FrameStack;
  GC::Allocator &Allocator;
  /// @}
  template <typename T> static T get(const Value &V) noexcept {
    if constexpr (std::is_same_v<detail::remove_cvref_t<T>, Value>) {
      return V;
    } else {
      return V.get<T>();
    }
  }
  template <typename T> static void emplace(Value &V, T &&Val) noexcept {
    using U = detail::remove_cvref_t<T>;
    if constexpr (std::is_same_v<U, Value>) {
      V = std::forward<T>(Val);
    } else {
      if constexpr (sizeof(U) < sizeof(Value)) {
        // Clear the slot before writing a value narrower than ValVariant. The
        // conservative GC scans the value stack and reads the high 64-bit word
        // of each slot as a candidate pointer, so bits left behind by a
        // previously stored reference (e.g. overwriting a ref with the i32
        // result of ref.is_null, or a local.set storing a number over a ref)
        // would be treated as a live root and retain the dead object.
        // ValVariant is trivially copyable, so zeroing the raw storage first is
        // well-defined.
        std::memset(static_cast<void *>(&V), 0, sizeof(V));
      }
      V.emplace<U>(std::forward<T>(Val));
    }
  }
};

} // namespace Runtime
} // namespace WasmEdge
