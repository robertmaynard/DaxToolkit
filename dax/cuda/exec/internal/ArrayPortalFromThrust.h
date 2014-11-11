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
#ifndef __dax_cuda_exec_internal_ArrayPortalFromThrust_h
#define __dax_cuda_exec_internal_ArrayPortalFromThrust_h

#include <dax/Types.h>

#include <iterator>

namespace dax {
namespace cuda {
namespace exec {
namespace internal {

class ArrayPortalFromThrustBase {};

/// This templated implementation of an ArrayPortal allows you to adapt a pair
/// of begin/end iterators to an ArrayPortal interface.
///
template<typename T>
class ArrayPortalFromThrust : public ArrayPortalFromThrustBase
{
public:
  typedef T ValueType;
  typedef typename thrust::system::cuda::pointer< T > PointerType;
  typedef T* IteratorType;

  DAX_EXEC_CONT_EXPORT ArrayPortalFromThrust() {  }

  DAX_CONT_EXPORT
  ArrayPortalFromThrust(PointerType begin, PointerType end)
    : BeginIterator( begin ),
      EndIterator( end  )
      {  }

  /// Copy constructor for any other ArrayPortalFromThrust with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalFromThrust(const ArrayPortalFromThrust<OtherT> &src)
    : BeginIterator(src.BeginIterator),
      EndIterator(src.EndIterator)
  {  }

  template<typename OtherT>
  DAX_EXEC_CONT_EXPORT
  ArrayPortalFromThrust<T> &operator=(
      const ArrayPortalFromThrust<OtherT> &src)
  {
    this->BeginIterator = src.BeginIterator;
    this->EndIterator = src.EndIterator;
    return *this;
  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    // Not using std::distance because on CUDA it cannot be used on a device.
    return (this->EndIterator - this->BeginIterator);
  }

  DAX_EXEC_EXPORT
  ValueType Get(dax::Id index) const {
    return *this->IteratorAt(index);
  }

  DAX_EXEC_EXPORT
  void Set(dax::Id index, ValueType value) const {
    *this->IteratorAt(index) = value;
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const { return this->BeginIterator.get(); }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const { return this->EndIterator.get(); }

private:
  PointerType BeginIterator;
  PointerType EndIterator;

  DAX_EXEC_EXPORT
  PointerType IteratorAt(dax::Id index) const {
    // Not using std::advance because on CUDA it cannot be used on a device.
    return (this->BeginIterator + index);
  }
};

template<typename T>
class ConstArrayPortalFromThrust : public ArrayPortalFromThrustBase
{
public:

  typedef T ValueType;
  typedef typename thrust::system::cuda::pointer< T > PointerType;
  typedef const T* IteratorType;

  DAX_EXEC_CONT_EXPORT ConstArrayPortalFromThrust() {  }

  DAX_CONT_EXPORT
  ConstArrayPortalFromThrust(const PointerType begin, const PointerType end)
    : BeginIterator( begin ),
      EndIterator( end )
      {  }

  /// Copy constructor for any other ConstArrayPortalFromThrust with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<typename OtherT>
  DAX_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrust(const ConstArrayPortalFromThrust<OtherT> &src)
    : BeginIterator(src.BeginIterator),
      EndIterator(src.EndIterator)
  {  }

  template<typename OtherT>
  DAX_EXEC_CONT_EXPORT
  ConstArrayPortalFromThrust<T> &operator=(
      const ConstArrayPortalFromThrust<OtherT> &src)
  {
    this->BeginIterator = src.BeginIterator;
    this->EndIterator = src.EndIterator;
    return *this;
  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    // Not using std::distance because on CUDA it cannot be used on a device.
    return (this->EndIterator - this->BeginIterator);
  }

  DAX_EXEC_EXPORT
  ValueType Get(dax::Id index) const {
    return *this->IteratorAt(index);
  }

  DAX_EXEC_EXPORT
  void Set(dax::Id index, ValueType value) const {
    *this->IteratorAt(index) = value;
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const { return this->BeginIterator.get(); }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const { return this->EndIterator.get(); }

private:
  PointerType BeginIterator;
  PointerType EndIterator;

  DAX_EXEC_EXPORT
  PointerType IteratorAt(dax::Id index) const {
    // Not using std::advance because on CUDA it cannot be used on a device.
    return (this->BeginIterator + index);
  }
};

}
}
}
} // namespace dax::cuda::exec::internal


#endif //__dax_cuda_exec_internal_ArrayPortalFromThrust_h
