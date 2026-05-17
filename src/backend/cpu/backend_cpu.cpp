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

#include "pypto/backend/cpu/backend_cpu.h"

#include <map>
#include <vector>

#include "pypto/backend/cpu/backend_cpu_handler.h"
#include "pypto/backend/common/soc.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/pipe.h"

namespace pypto {
namespace backend {

namespace {

const SoC& CreateCPUSoC() {
  static SoC soc = []() {
    // Single CPU compute core with DDR only
    Core cpu_core(ir::CoreType::VECTOR, {
                                            Mem(ir::MemorySpace::DDR, 1ULL << 40, 64),  // 1 TB DDR
                                        });

    Cluster cpu_cluster(cpu_core, 1);
    Die cpu_die(cpu_cluster, 1);

    // DDR is the only memory node
    std::map<ir::MemorySpace, std::vector<ir::MemorySpace>> mem_graph;
    mem_graph[ir::MemorySpace::DDR] = {};

    return SoC(cpu_die, 1, std::move(mem_graph));
  }();
  return soc;
}

}  // namespace

BackendCPU::BackendCPU() : Backend(CreateCPUSoC()) {}

BackendCPU& BackendCPU::Instance() {
  static BackendCPU instance;
  return instance;
}

const BackendHandler* BackendCPU::GetHandler() const {
  return &CPUBackendHandler::Instance();
}

}  // namespace backend
}  // namespace pypto
