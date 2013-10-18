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

#include <dax/cont/testing/TestingGridGenerator.h>
#include <dax/cont/testing/Testing.h>

#include <dax/worklet/MarchingCubesMapReduce.h>

#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/TypeTraits.h>

#include <dax/cont/ArrayContainerControlBasic.h>
#include <dax/cont/ArrayHandle.h>
#include <dax/cont/GenerateKeysValues.h>
#include <dax/cont/ReduceKeysValues.h>
#include <dax/cont/Scheduler.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/UnstructuredGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/cont/testing/Testing.h>
#include <vector>


namespace {
const dax::Id DIM = 26;
const dax::Id ISOVALUE = 70;

//-----------------------------------------------------------------------------
struct TestMarchingCubesMapReduceWorklet
{
  typedef dax::cont::ArrayContainerControlTagBasic ArrayContainer;
  typedef DAX_DEFAULT_DEVICE_ADAPTER_TAG DeviceAdapter;

  typedef dax::CellTagTriangle CellType;

  typedef dax::cont::UnstructuredGrid<
      CellType,ArrayContainer,ArrayContainer,DeviceAdapter>
      UnstructuredGridType;

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  template<class InputGridType>
  void operator()(const InputGridType&) const
    {
    dax::cont::testing::TestGrid<InputGridType,ArrayContainer,DeviceAdapter>
        inGrid(DIM);
    UnstructuredGridType outGrid;

    dax::Vector3 trueGradient = dax::make_Vector3(1.0, 1.0, 1.0);
    dax::Id numPoints = inGrid->GetNumberOfPoints();
    std::vector<dax::Scalar> field(numPoints);
    for (dax::Id pointIndex = 0; pointIndex < numPoints; pointIndex++)
      {
      dax::Vector3 coordinates = inGrid.GetPointCoordinates(pointIndex);
      field[pointIndex] = dax::dot(coordinates, trueGradient);
      }

    dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter>
        fieldHandle = dax::cont::make_ArrayHandle(field,
                                                  ArrayContainer(),
                                                  DeviceAdapter());

    const dax::Scalar isoValue = ISOVALUE;

    try
      {
      typedef dax::cont::ArrayHandle<dax::Id, ArrayContainer, DeviceAdapter>
      ClassifyResultType;

      //construct the scheduler that will execute all the worklets
      dax::cont::Scheduler<DeviceAdapter> scheduler;

      //construct the two worklets that will be used to do the marching cubes
      dax::worklet::MarchingCubesClassify classifyWorklet(isoValue);
      dax::worklet::MarchingCubesGenerate generateWorklet(isoValue);
      dax::worklet::MarchingCubesInterpolate interpolateWorklet;

      //run the first step
      ClassifyResultType classification;
      scheduler.Invoke(classifyWorklet,
                       inGrid.GetRealGrid(),
                       fieldHandle,
                       classification);

      //construct the mapping of key->value where key is two edge ids
      //that construct the new point and the value is the weight
      dax::cont::ArrayHandle<dax::Id2,ArrayContainer,DeviceAdapter> keyHandle;
      dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter> valueHandle;

      dax::cont::GenerateKeysValues< dax::worklet::MarchingCubesGenerate,
                                     ClassifyResultType >
          generateKeys( classification, generateWorklet );

      //run the Map step
      scheduler.Invoke(generateKeys,
                       inGrid.GetRealGrid(),
                       fieldHandle,
                       keyHandle,
                       valueHandle);

      //construct the reduce step
      dax::cont::ReduceKeysValues< dax::worklet::MarchingCubesInterpolate,
                             dax::cont::ArrayHandle<dax::Id2,
                             ArrayContainer, DeviceAdapter> >
                    reduceKeys(keyHandle, interpolateWorklet);
      dax::cont::ArrayHandle<dax::Scalar,ArrayContainer,DeviceAdapter> reducedValues;
      scheduler.Invoke(reduceKeys, valueHandle, reducedValues);

    }
    catch (dax::cont::ErrorControl error)
      {
      std::cout << "Got error: " << error.GetMessage() << std::endl;
      DAX_TEST_ASSERT(true==false,error.GetMessage());
      }
    }
};

//-----------------------------------------------------------------------------
void TestMarchingCubesMapReduce()
  {
  dax::cont::testing::GridTesting::TryAllGridTypes(
        TestMarchingCubesMapReduceWorklet(),
        dax::testing::Testing::CellCheckHexahedron());
  }
} // Anonymous namespace

//-----------------------------------------------------------------------------
int UnitTestWorkletMarchingCubesMapReduce(int, char *[])
{
  return dax::cont::testing::Testing::Run(TestMarchingCubesMapReduce);
}
