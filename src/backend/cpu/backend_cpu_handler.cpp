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

#include "pypto/backend/cpu/backend_cpu_handler.h"

#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

const CPUBackendHandler& CPUBackendHandler::Instance() {
  static const CPUBackendHandler instance;
  return instance;
}

ir::TileView CPUBackendHandler::BuildCrossCoreTransferView(
    ir::MemorySpace /*dest_ms*/, const ir::TileView& original_view) const {
  return original_view;
}

}  // namespace backend
}  // namespace pypto
