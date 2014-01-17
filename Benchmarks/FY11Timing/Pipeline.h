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
#include <stdio.h>
#include <iostream>

#include <dax/cont/ArrayHandle.h>
#include <dax/cont/ArrayHandleTransform.h>
#include <dax/cont/DispatcherMapCell.h>
#include <dax/cont/DispatcherMapField.h>
#include <dax/cont/Timer.h>
#include <dax/cont/UniformGrid.h>
#include <dax/cont/VectorOperations.h>

#include <dax/worklet/CellGradient.h>
#include <dax/worklet/Cosine.h>
#include <dax/worklet/Magnitude.h>
#include <dax/worklet/Sine.h>
#include <dax/worklet/Square.h>

#include <vector>

#define MAKE_STRING2(x) #x
#define MAKE_STRING1(x) MAKE_STRING2(x)
#define DEVICE_ADAPTER MAKE_STRING1(DAX_DEFAULT_DEVICE_ADAPTER_TAG)

namespace functors
{
struct Mag
{
  DAX_EXEC_CONT_EXPORT
  dax::Scalar operator()(dax::Vector3 inValue) const
  {
    return dax::math::Magnitude(inValue);
  }
};

struct Sine
{
  template<class ValueType>
  DAX_EXEC_CONT_EXPORT
  ValueType operator()(const ValueType &inValue) const
  {
    return dax::math::Sin(inValue);
  }
};

struct Square
{
  template<class ValueType>
  DAX_EXEC_CONT_EXPORT
  ValueType operator()(const ValueType &inValue) const
  {
    return inValue * inValue;
  }
};

}

namespace
{

class CheckValid {
public:
  CheckValid() : Valid(true) { }
  operator bool() { return this->Valid; }
  void operator()(dax::Scalar value) {
    if ((value < -1) || (value > 2)) { this->Valid = false; }
  }
private:
  bool Valid;
};

void PrintScalarValue(dax::Scalar value)
{
  std::cout << " " << value;
}

template<class IteratorType>
void PrintCheckValues(IteratorType begin, IteratorType end)
{
  typedef typename std::iterator_traits<IteratorType>::value_type VectorType;

  dax::Id index = 0;
  for (IteratorType iter = begin; iter != end; iter++)
    {
    VectorType vector = *iter;
    if (index < 20)
      {
      std::cout << index << ":";
      dax::cont::VectorForEach(vector, PrintScalarValue);
      std::cout << std::endl;
      }
    else
      {
      CheckValid isValid;
      dax::cont::VectorForEach(vector, isValid);
      if (!isValid)
        {
        std::cout << "*** Encountered bad value." << std::endl;
        std::cout << index << ":";
        dax::cont::VectorForEach(vector, PrintScalarValue);
        std::cout << std::endl;
        exit(1);
        }
      }

    index++;
    }
}

template<typename T, class Container, class Device>
void PrintCheckValues(const dax::cont::ArrayHandle<T,Container,Device> &array)
{
  PrintCheckValues(array.GetPortalConstControl().GetIteratorBegin(),
                   array.GetPortalConstControl().GetIteratorEnd());
}

void PrintResults(int pipeline, double time)
{
  std::cout << "Elapsed time: " << time << " seconds." << std::endl;
  std::cout << "CSV," DEVICE_ADAPTER ","
            << pipeline << "," << time << std::endl;
}

void RunPipeline1(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 1: Magnitude -> Gradient" << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;

  dax::cont::ArrayHandle<dax::Vector3> results;

  dax::cont::Timer<> timer;

  dax::cont::DispatcherMapField< dax::worklet::Magnitude >().Invoke(
        grid.GetPointCoordinates(),
        intermediate1);

  dax::cont::DispatcherMapCell< dax::worklet::CellGradient >().Invoke(
        grid,
        grid.GetPointCoordinates(),
        intermediate1,
        results);

  double time = timer.GetElapsedTime();

  PrintCheckValues(results);
  PrintResults(1, time);
}

void RunPipeline2(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 2: Magnitude->Gradient->Sine->Square->Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::ArrayHandle<dax::Vector3> intermediate2;
  dax::cont::ArrayHandle<dax::Vector3> intermediate3;

  dax::cont::ArrayHandle<dax::Vector3> results;

  dax::cont::Timer<> timer;

  dax::cont::DispatcherMapField<dax::worklet::Magnitude>().Invoke(
        grid.GetPointCoordinates(),
        intermediate1);

  dax::cont::DispatcherMapCell< dax::worklet::CellGradient >().Invoke(
        grid,
        grid.GetPointCoordinates(),
        intermediate1,
        intermediate2);

  intermediate1.ReleaseResources();


  dax::cont::DispatcherMapField< dax::worklet::Sine >().Invoke(intermediate2,
                                                               intermediate3);
  dax::cont::DispatcherMapField< dax::worklet::Square>().Invoke(intermediate3,
                                                                intermediate2);
  intermediate3.ReleaseResources();
  dax::cont::DispatcherMapField< dax::worklet::Cosine>().Invoke(intermediate2,
                                                                results);
  double time = timer.GetElapsedTime();

  PrintCheckValues(results);

  PrintResults(2, time);
}

void RunPipeline3(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running pipeline 3: Magnitude -> Sine -> Square -> Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> intermediate1;
  dax::cont::ArrayHandle<dax::Scalar> intermediate2;

  dax::cont::ArrayHandle<dax::Scalar> results;

  dax::cont::Timer<> timer;

  dax::cont::DispatcherMapField<dax::worklet::Magnitude>().Invoke(
        grid.GetPointCoordinates(),
        intermediate1);

  dax::cont::DispatcherMapField< dax::worklet::Sine >().Invoke(intermediate1,
                                                               intermediate2);
  dax::cont::DispatcherMapField< dax::worklet::Square>().Invoke(intermediate2,
                                                                intermediate1);
  intermediate2.ReleaseResources();
  dax::cont::DispatcherMapField< dax::worklet::Cosine>().Invoke(intermediate1,
                                                                results);

  double time = timer.GetElapsedTime();

  PrintCheckValues(results);

  PrintResults(3, time);
}

void RunPipeline4(const dax::cont::UniformGrid<> &grid)
{
  std::cout << "Running Fused Pipeline 4: Magnitude -> Sine -> Square -> Cosine"
            << std::endl;

  dax::cont::ArrayHandle<dax::Scalar> results;

  dax::cont::Timer<> timer;

  //fuse the first three worklets into the transform handle instead of
  //explicitly calling each of them, only the cosine worklet will be called
  typedef dax::cont::ArrayHandleTransform< dax::Scalar,
          dax::cont::UniformGrid<>::PointCoordinatesType,
          ::functors::Mag > MagnitudeHandle;

  typedef dax::cont::ArrayHandleTransform< dax::Scalar,
          MagnitudeHandle,
          ::functors::Sine > SineFusedHandle;

  typedef dax::cont::ArrayHandleTransform< dax::Scalar,
          SineFusedHandle,
          ::functors::Square > SquareFusedHandle;

  MagnitudeHandle mag(grid.GetPointCoordinates());

  SineFusedHandle sine(mag);
  SquareFusedHandle fused(sine);

  dax::cont::DispatcherMapField< dax::worklet::Cosine>().Invoke(fused,
                                                                results);

  double time = timer.GetElapsedTime();

  PrintCheckValues(results);


  PrintResults(4, time);
}

} // Anonymous namespace

