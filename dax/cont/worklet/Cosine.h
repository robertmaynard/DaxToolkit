//=============================================================================
//
//  Copyright (c) Kitware, Inc.
//  All rights reserved.
//  See LICENSE.txt for details.
//
//  This software is distributed WITHOUT ANY WARRANTY; without even
//  the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
//  PURPOSE.  See the above copyright notice for more information.
//
//  Copyright 2012 Sandia Corporation.
//  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
//  the U.S. Government retains certain rights in this software.
//
//=============================================================================
#ifndef __dax_cont_worklet_Cosine_h
#define __dax_cont_worklet_Cosine_h

// TODO: This should be auto-generated.

#include <dax/Types.h>
#include <dax/exec/Field.h>
#include <dax/exec/WorkMapField.h>
#include <dax/exec/internal/ExecutionAdapter.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/DeviceAdapter.h>
#include <dax/cont/ErrorControlBadValue.h>
#include <dax/cont/internal/ExecutionPackageField.h>
#include <dax/cont/internal/ExecutionPackageGrid.h>

#include <Worklets/Cosine.worklet>

namespace dax {
namespace exec {
namespace internal {
namespace kernel {

template<class CellType, typename FieldType, class ExecAdapter>
struct CosineParameters
{
  typename CellType::template GridStructures<ExecAdapter>::TopologyType grid;
  dax::exec::FieldIn<FieldType, ExecAdapter> inField;
  dax::exec::FieldOut<FieldType, ExecAdapter> outField;
};

template<class CellType, typename FieldType, class ExecAdapter>
struct Cosine
{
  DAX_EXEC_EXPORT void operator()(
      CosineParameters<CellType, FieldType, ExecAdapter> &parameters,
      dax::Id index,
      const ExecAdapter &execAdapter) const
  {
    dax::exec::WorkMapField<CellType, ExecAdapter>
        work(parameters.grid, index, execAdapter);
    dax::worklet::Cosine(work,
                         parameters.inField,
                         parameters.outField);
  }
};

}
}
}
} // dax::exec::internal::kernel

namespace dax {
namespace cont {
namespace worklet {

template<class GridType,
         typename FieldType,
         class Container,
         class Adapter>
DAX_CONT_EXPORT void Cosine(
    const GridType &grid,
    const dax::cont::ArrayHandle<FieldType, Container, Adapter> &inHandle,
    dax::cont::ArrayHandle<FieldType, Container, Adapter> &outHandle)
{
  dax::Id fieldSize;
  if (inHandle.GetNumberOfValues() == grid.GetNumberOfPoints())
    {
    fieldSize = grid.GetNumberOfPoints();
    }
  else if (inHandle.GetNumberOfValues() == grid.GetNumberOfCells())
    {
    fieldSize = grid.GetNumberOfCells();
    }
  else
    {
    throw dax::cont::ErrorControlBadValue(
          "Number of array entries neither cells nor points.");
    }

  typedef dax::exec::internal::ExecutionAdapter<Container,Adapter> ExecAdapter;

  typedef typename GridType::ExecutionTopologyStruct ExecutionTopologyType;
  ExecutionTopologyType execTopology
      = dax::cont::internal::ExecutionPackageGrid(grid);

  dax::exec::FieldIn<FieldType, ExecAdapter> fieldIn
      = dax::cont::internal::ExecutionPackageField<dax::exec::FieldIn>(
        inHandle, fieldSize);

  dax::exec::FieldOut<FieldType, ExecAdapter> fieldOut
      = dax::cont::internal::ExecutionPackageField<dax::exec::FieldOut>(
        outHandle, fieldSize);

  typedef typename GridType::CellType CellType;

  typedef dax::exec::internal::kernel::CosineParameters<
      CellType, FieldType, ExecAdapter> Parameters;

  Parameters parameters;
  parameters.grid = execTopology;
  parameters.inField = fieldIn;
  parameters.outField = fieldOut;

  dax::cont::internal::Schedule(
        dax::exec::internal::kernel::Cosine<CellType,FieldType,ExecAdapter>(),
        parameters,
        fieldSize,
        Container(),
        Adapter());
}

}
}
} //dax::cont::worklet

#endif //__dax_cont_worklet_Cosine_h
