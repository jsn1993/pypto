/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#ifndef PYPTO_BACKEND_CPU_BACKEND_CPU_HANDLER_H_
#define PYPTO_BACKEND_CPU_BACKEND_CPU_HANDLER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

class CPUBackendHandler : public BackendHandler {
 public:
  static const CPUBackendHandler& Instance();

  // ---- Codegen hooks (not used — CPU codegen is Python, not PTO) ----
  [[nodiscard]] std::string GetPtoTargetArch() const override { return "cpu"; }
  [[nodiscard]] std::string GetLaunchSpecCoreCountMethod() const override { return ""; }
  [[nodiscard]] std::string GetDefaultSimPlatform() const override { return "cpu"; }
  [[nodiscard]] std::vector<std::string> GetExtraPtoasFlags() const override { return {}; }

  // ---- Pass workarounds — none needed on CPU ----
  [[nodiscard]] bool RequiresGMPipeBuffer() const override { return false; }
  [[nodiscard]] bool RequiresSplitLoadTpopWorkaround() const override { return false; }
  [[nodiscard]] bool RequiresVtoCFractalAdapt() const override { return false; }
  [[nodiscard]] bool RequiresRuntimeSubblockBridge() const override { return false; }
  [[nodiscard]] bool RequiresNoSplitDualAivDispatch() const override { return false; }

  // ---- No cross-core transfers on CPU ----
  [[nodiscard]] ir::TileView BuildCrossCoreTransferView(
      ir::MemorySpace dest_ms, const ir::TileView& original_view) const override;

  // ---- Performance thresholds — CPU cache line is 64 bytes ----
  [[nodiscard]] uint32_t GetGmAccessGranularityBytes() const override { return 64; }
  [[nodiscard]] uint32_t GetL2CacheLineBytes() const override { return 64; }
  [[nodiscard]] uint32_t GetRecommendedInnermostDimBytes() const override { return 64; }

  // ---- L0 capacities — unlimited (AutoTileMatmulL0 skipped on CPU) ----
  [[nodiscard]] uint32_t GetL0aCapacityBytes() const override { return 1U << 30; }
  [[nodiscard]] uint32_t GetL0bCapacityBytes() const override { return 1U << 30; }
  [[nodiscard]] uint32_t GetL0cCapacityBytes() const override { return 1U << 30; }

 private:
  CPUBackendHandler() = default;
};

}  // namespace backend
}  // namespace pypto

#endif  // PYPTO_BACKEND_CPU_BACKEND_CPU_HANDLER_H_
