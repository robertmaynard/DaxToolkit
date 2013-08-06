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

#include "ArgumentsParser.h"

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/GenerateKeysValues.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/worklet/Magnitude.h>
#include <dax/worklet/MarchingCubesMapReduce.h>

#include <iostream>
#include <vector>
#include <fstream>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace
{

dax::Scalar ISOVALUE = 5;

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

void RunDAXPipeline(const dax::cont::UniformGrid<> &grid, int pipeline)
{
  std::cout << "Running pipeline 1: Magnitude -> MarchingCubes" << std::endl;

  dax::cont::UnstructuredGrid<dax::CellTagTriangle> outGrid;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::Scheduler<> schedule;

  schedule.Invoke(dax::worklet::Magnitude(),
                  grid.GetPointCoordinates(),
                  intermediate1);

  dax::cont::Timer<> timer;

  //schedule marching cubes worklet generate step
  typedef dax::cont::GenerateKeysValues<dax::worklet::MarchingCubesGenerate> GenerateKV;
  typedef GenerateKV::OutputCountType  OutputCountType;

  typedef dax::cont::ReduceKeysValues< dax::worklet::MarchingCubesInterpolate,
                             dax::cont::ArrayHandle<dax::Id2> > ReduceKV;

  dax::worklet::MarchingCubesClassify classifyWorklet(ISOVALUE);
  dax::worklet::MarchingCubesGenerate generateWorklet(ISOVALUE);


  //run the first step
  OutputCountType keyCountsPerCell;
  schedule.Invoke(classifyWorklet, grid, intermediate1, keyCountsPerCell);

  //construct the mapping of key->value where key is two edge ids
  //that construct the new point and the value is the weight
  dax::cont::ArrayHandle<dax::Id2> keyHandle;
  dax::cont::ArrayHandle<dax::Scalar> valueHandle;
  GenerateKV generate(keyCountsPerCell,generateWorklet);

  //run the second step
  schedule.Invoke(generate,
                   grid, intermediate1, keyHandle, valueHandle);


  dax::cont::ArrayHandle<dax::Scalar> reducedValues;
  schedule.Invoke(ReduceKV(keyHandle), valueHandle, reducedValues);

  double time = timer.GetElapsedTime();

  PrintResults(1, time);
}


} // Anonymous namespace

