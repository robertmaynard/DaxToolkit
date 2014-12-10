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
#ifndef __dax_cuda_cont_internal_ArrayManagerExecutionThrustDevice_h
#define __dax_cuda_cont_internal_ArrayManagerExecutionThrustDevice_h

#include <dax/thrust/cont/internal/CheckThrustBackend.h>

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ErrorControlOutOfMemory.h>

// Disable GCC warnings we check Dax for but Thrust does not.
#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#endif // gcc version >= 4.6
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 2)
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif // gcc version >= 4.2
#endif // gcc && !CUDA

#include <thrust/system/cuda/memory.h>
#include <thrust/copy.h>

#if defined(__GNUC__) && !defined(DAX_CUDA)
#if (__GNUC__ >= 4) && (__GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif // gcc version >= 4.6
#endif // gcc && !CUDA

#include <dax/cuda/exec/internal/UninitializedAllocator.h>
#include <dax/cuda/exec/internal/ArrayPortalFromThrust.h>
#include <dax/cuda/exec/internal/ArrayPortalFromTexture.h>

#if DAX_THRUST_MAJOR_VERSION == 1 && DAX_THRUST_MINOR_VERSION >= 7
# ifndef DAX_USE_TEXTURE_MEM
#  define DAX_USE_TEXTURE_MEM
# endif
#endif

namespace dax {
namespace cuda {
namespace cont {
namespace internal {

/// \c ArrayManagerExecutionThrustDevice provides an implementation for a \c
/// ArrayManagerExecution class for a thrust device adapter that is designed
/// for the cuda backend which has separate memory spaces for host and device. This
/// implementation contains a ::thrust::system::cuda::vector to allocate and manage
/// the array.
///
/// This array manager should only be used with the cuda device adapter,
/// since in the future it will take advantage of texture memory and
/// the unique memory access patterns of cuda systems.
template<typename T, class ArrayContainerControlTag>
class ArrayManagerExecutionThrustDevice
{
  // typedef dax::Scalar T;
//we need a way to detect that we are using FERMI or lower and disable
//the usage of texture iterator. The __CUDA_ARCH__ define is only around
//for device code so that can't be used. I expect that we will have to devise
//some form of Try/Compile with CUDA or just offer this as an advanced CMake
//option. We could also try and see if a runtime switch is possible.
#ifdef DAX_USE_TEXTURE_MEM
  typedef ::dax::cuda::exec::internal::DaxTexObjInputIterator<T> TextureIteratorType;
#endif

public:
  typedef T ValueType;

  typedef dax::cont::internal
      ::ArrayContainerControl<ValueType, ArrayContainerControlTag>
      ContainerType;


  typedef dax::cuda::exec::internal::ArrayPortalFromThrust< T >
      PortalType;

#ifdef DAX_USE_TEXTURE_MEM
  typedef dax::cuda::exec::internal::ConstArrayPortalFromTexture< TextureIteratorType > PortalConstType;
#else
    typedef dax::cuda::exec::internal::ConstArrayPortalFromThrust< T > PortalConstType;
#endif

  DAX_CONT_EXPORT ArrayManagerExecutionThrustDevice():
    NumberOfValues(0),
    ArrayBegin(),
    ArrayEnd(),
    HaveTextureBound(false)
#ifdef DAX_USE_TEXTURE_MEM
    ,
    InputArrayIterator()
#endif
  {

  }

  ~ArrayManagerExecutionThrustDevice()
  {
  if(this->HaveTextureBound)
    {
    this->HaveTextureBound = false;
#ifdef DAX_USE_TEXTURE_MEM
    this->InputArrayIterator.UnbindTexture();
#endif
    }
  }

  /// Returns the size of the array.
  ///
  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const {
    return this->NumberOfValues;
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template<class PortalControl>
  DAX_CONT_EXPORT void LoadDataForInput(PortalControl arrayPortal)
  {
    try
      {
      this->NumberOfValues = arrayPortal.GetNumberOfValues();
      this->ArrayBegin = ::thrust::system::cuda::malloc<T>( static_cast<std::size_t>(this->NumberOfValues)  );
      this->ArrayEnd = this->ArrayBegin + this->NumberOfValues;

      ::thrust::copy(arrayPortal.GetIteratorBegin(),
                     arrayPortal.GetIteratorEnd(),
                     this->ArrayBegin);
      }
    catch (std::bad_alloc error)
      {
      throw dax::cont::ErrorControlOutOfMemory(error.what());
      }
  }

  /// Allocates the appropriate size of the array and copies the given data
  /// into the array.
  ///
  template<class PortalControl>
  DAX_CONT_EXPORT void LoadDataForInPlace(PortalControl arrayPortal)
  {
    //inplace is we are not allowed to have a texture mapping as textures are
    //read only
    try
      {
      //we allocate global memory and copy into it
      this->NumberOfValues = arrayPortal.GetNumberOfValues();
      this->ArrayBegin = ::thrust::system::cuda::malloc<T>( this->NumberOfValues  );
      this->ArrayEnd = this->ArrayBegin + this->NumberOfValues;

      ::thrust::copy(arrayPortal.GetIteratorBegin(),
                     arrayPortal.GetIteratorEnd(),
                     this->ArrayBegin);
      }
    catch (std::bad_alloc error)
      {
      throw dax::cont::ErrorControlOutOfMemory(error.what());
      }
  }

  /// Allocates the array to the given size.
  ///
  DAX_CONT_EXPORT void AllocateArrayForOutput(
      ContainerType &daxNotUsed(container),
      dax::Id numberOfValues)
  {
    if(this->NumberOfValues > 0)
      {
      ::thrust::system::cuda::free( this->ArrayBegin  );
      }
    this->NumberOfValues = numberOfValues;
    this->ArrayBegin = ::thrust::system::cuda::malloc<T>( this->NumberOfValues  );
    this->ArrayEnd = this->ArrayBegin + numberOfValues;
  }

  /// Copies the data currently in the device array into the given iterators.
  /// Although the iterator is supposed to be from the control environment,
  /// thrust can generally handle iterators for a device as well.
  ///
  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    ::thrust::copy(this->ArrayBegin, this->ArrayEnd, dest);
  }

  /// Allocates enough space in \c controlArray and copies the data in the
  /// device vector into it.
  ///
  DAX_CONT_EXPORT void RetrieveOutputData(ContainerType &controlArray) const
  {
    controlArray.Allocate(this->NumberOfValues);
    this->CopyInto(controlArray.GetPortal().GetIteratorBegin());
  }

  /// Resizes the device vector.
  ///
  DAX_CONT_EXPORT void Shrink(dax::Id numberOfValues)
  {
    // The operation will succeed even if this assertion fails, but this
    // is still supposed to be a precondition to Shrink.
    DAX_ASSERT_CONT(numberOfValues <= static_cast<dax::Id>(this->Array.size()));
    this->NumberOfValues = numberOfValues;
    this->ArrayEnd = this->ArrayBegin + this->NumberOfValues;
  }

  DAX_CONT_EXPORT PortalType GetPortal()
  {
    return PortalType(this->ArrayBegin, this->ArrayEnd);
  }

  DAX_CONT_EXPORT PortalConstType GetPortalConst() const
  {
#ifdef DAX_USE_TEXTURE_MEM
    if(!this->HaveTextureBound)
      {
      this->HaveTextureBound = true;
      this->InputArrayIterator.BindTexture(ArrayBegin,NumberOfValues);
      }

    //if we have a texture iterator bound use that
    return PortalConstType(this->InputArrayIterator, this->NumberOfValues );
#else
    return PortalConstType(this->ArrayBegin, this->ArrayEnd);
#endif
  }


  /// Frees all memory.
  ///
  DAX_CONT_EXPORT void ReleaseResources() {
  if(this->HaveTextureBound)
    {
    this->HaveTextureBound = false;
#ifdef DAX_USE_TEXTURE_MEM
    this->InputArrayIterator.UnbindTexture();
#endif
    }
    ::thrust::system::cuda::free( this->ArrayBegin  );
    this->ArrayBegin = ::thrust::system::cuda::pointer<ValueType>();
    this->ArrayEnd = ::thrust::system::cuda::pointer<ValueType>();
  }

private:
  // Not implemented
  ArrayManagerExecutionThrustDevice(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);
  void operator=(
      ArrayManagerExecutionThrustDevice<T, ArrayContainerControlTag> &);

  dax::Id NumberOfValues;
  ::thrust::system::cuda::pointer<ValueType> ArrayBegin;
  ::thrust::system::cuda::pointer<ValueType> ArrayEnd;
  mutable bool HaveTextureBound;
#ifdef DAX_USE_TEXTURE_MEM
  mutable TextureIteratorType InputArrayIterator;
#endif
};


}
}
}
} // namespace dax::cuda::cont::internal

#endif // __dax_cuda_cont_internal_ArrayManagerExecutionThrustDevice_h
