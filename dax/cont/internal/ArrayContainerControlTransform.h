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
#ifndef __dax_cont_internal_ArrayHandleTransform_h
#define __dax_cont_internal_ArrayHandleTransform_h

#include <dax/cont/ArrayContainerControl.h>
#include <dax/cont/ErrorControlInternal.h>
#include <dax/cont/internal/ArrayTransfer.h>
#include <dax/cont/internal/IteratorFromArrayPortal.h>

namespace dax {
namespace cont {
namespace internal {


//simple container for the array handle and functor
//so we can get them inside the array transfer class
template< class T, class ArrayHandleType, class FunctorType>
struct ArrayPortalConstTransform
{
  typedef T ValueType;
  typedef typename ArrayHandleType::PortalConstControl PortalType;

  DAX_CONT_EXPORT
  ArrayPortalConstTransform():
  Handle(),
  Portal(),
  Functor()
  { }

  DAX_CONT_EXPORT
  ArrayPortalConstTransform(ArrayHandleType handle, FunctorType f):
  Handle(handle),
  Portal(handle.GetPortalConstControl()),
  Functor(f)
  { }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    return this->Handle.GetNumberOfValues();
  }

  DAX_CONT_EXPORT
  ValueType Get(dax::Id index) const {
    // ArrayHandleType& a = 'a';
    // PortalType portal = this->Handle.GetPortalConstControl();

    return this->Functor(1);
  }


  typedef dax::cont::internal::IteratorFromArrayPortal<
      ArrayPortalConstTransform <
                    T, ArrayHandleType, FunctorType> > IteratorType;

  DAX_CONT_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_CONT_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->GetNumberOfValues());
  }


  ArrayHandleType Handle;
  PortalType Portal;
  FunctorType Functor;
};

template <class ValueType_,
          class PortalType_,
          class FunctorType_>
class ArrayPortalExecTransform
{
public:
  typedef PortalType_ PortalType;
  typedef ValueType_ ValueType;
  typedef FunctorType_ FunctorType;

  DAX_CONT_EXPORT
  ArrayPortalExecTransform() :
  Portal(),
  NumberOfValues(0),
  Functor()
  {  }

  DAX_CONT_EXPORT
  ArrayPortalExecTransform(const PortalType& portal, dax::Id size, FunctorType f ) :
  Portal(portal),
  NumberOfValues(size),
  Functor(f)
  {  }

  /// Copy constructor for any other ArrayPortalExecTransform with an iterator
  /// type that can be copied to this iterator type. This allows us to do any
  /// type casting that the iterators do (like the non-const to const cast).
  ///
  template<class OtherV, class OtherP, class OtherF>
  DAX_CONT_EXPORT
  ArrayPortalExecTransform(const ArrayPortalExecTransform<OtherV,OtherP,OtherF> &src)
    : Portal(src.GetPortal()),
      NumberOfValues(src.GetNumberOfValues()),
      Functor(src.GetFunctor())

  {  }

  DAX_EXEC_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    return this->NumberOfValues;
  }

  DAX_EXEC_CONT_EXPORT
  ValueType Get(dax::Id index) const{
    return this->Functor(this->Portal.Get(index));
  }

  typedef dax::cont::internal::IteratorFromArrayPortal< ArrayPortalExecTransform <
                    ValueType, PortalType, FunctorType> > IteratorType;

  DAX_EXEC_EXPORT
  IteratorType GetIteratorBegin() const
  {
    return IteratorType(*this);
  }

  DAX_EXEC_EXPORT
  IteratorType GetIteratorEnd() const
  {
    return IteratorType(*this, this->GetNumberOfValues());
  }

  DAX_CONT_EXPORT
  PortalType &GetPortal() { return this->Portal; }
  DAX_CONT_EXPORT
  const PortalType &GetPortal() const { return this->Portal; }

  DAX_CONT_EXPORT
  FunctorType &GetFunctor() { return this->Portal; }
  DAX_CONT_EXPORT
  const FunctorType &GetFunctor() const { return this->Portal; }

private:
  PortalType Portal;
  dax::Id NumberOfValues;
  FunctorType Functor;
};


template<class ValueType, class HandleType, class FunctorType>
struct ArrayContainerControlTagTransform { };

/// A convenience class that provides a typedef to the appropriate tag for
/// a counting array container.
template<typename ValueType, typename ArrayHandleType, typename FunctorType>
struct ArrayHandleTransformTraits
{
  typedef dax::cont::internal::ArrayContainerControlTagTransform<
                                                ValueType,
                                                ArrayHandleType,
                                                FunctorType > Tag;
  typedef dax::cont::internal::ArrayContainerControl<
          ValueType, Tag > ContainerType;

};


template<typename T, class ArrayHandleType, class FunctorType>
class ArrayContainerControl<
    T,
    ArrayContainerControlTagTransform<T,ArrayHandleType,FunctorType> >
{
public:

  typedef T ValueType;
  typedef ArrayPortalConstTransform<T,ArrayHandleType,FunctorType> PortalType;
  typedef PortalType PortalConstType;

public:
  DAX_CONT_EXPORT
  ArrayContainerControl() {
  }


  DAX_CONT_EXPORT
  PortalType GetPortal() {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }

  DAX_CONT_EXPORT
  PortalConstType GetPortalConst() const {
    throw dax::cont::ErrorControlBadValue(
          "Transform container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }

  DAX_CONT_EXPORT
  dax::Id GetNumberOfValues() const {
    throw dax::cont::ErrorControlBadValue(
          "Transform container does not store array portal.  "
          "Perhaps you did not set the ArrayPortal when "
          "constructing the ArrayHandle.");
  }

  DAX_CONT_EXPORT
  void Allocate(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlInternal(
      "The allocate method for the transform control array container should "
      "never have been called. The allocate is generally only called by "
      "the execution array manager, and the array transfer for the transform "
      "container should prevent the execution array manager from being "
      "directly used.");
  }

  DAX_CONT_EXPORT
  void Shrink(dax::Id daxNotUsed(numberOfValues)) {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }

  DAX_CONT_EXPORT
  void ReleaseResources() {
    throw dax::cont::ErrorControlBadValue("Transform arrays are read-only.");
  }
};


template<typename T,
         class ArrayHandleType,
         class FunctorType,
         class DeviceAdapterTag>
class ArrayTransfer<
    T,
    ArrayContainerControlTagTransform< T, ArrayHandleType, FunctorType >,
    DeviceAdapterTag>
{
private:
  typedef ArrayContainerControlTagTransform< T, ArrayHandleType, FunctorType>
      ArrayContainerControlTag;
  typedef dax::cont::internal::ArrayContainerControl<T,ArrayContainerControlTag>
      ContainerType;

public:
  typedef T ValueType;

  typedef typename ContainerType::PortalType PortalControl;
  typedef typename ContainerType::PortalConstType PortalConstControl;

  typedef ArrayPortalExecTransform< ValueType,
                  typename ArrayHandleType::PortalExecution,
                  FunctorType> PortalExecution;

  typedef ArrayPortalExecTransform< ValueType,
                  typename ArrayHandleType::PortalConstExecution,
                  FunctorType> PortalConstExecution;


  DAX_CONT_EXPORT
  ArrayTransfer() :
    PortalValid(false),
    NumberOfValues(0),
    Portal(),
    InputPortal()
  {
  }

  DAX_CONT_EXPORT dax::Id GetNumberOfValues() const {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->NumberOfValues;
  }

  DAX_CONT_EXPORT void LoadDataForInput(PortalConstControl portal)
  {
    this->InputPortal = portal;

    typename ArrayHandleType::PortalConstExecution tmpInput =
                         this->InputPortal.Handle.PrepareForInput();
    this->NumberOfValues = this->InputPortal.Handle.GetNumberOfValues();

    this->Portal = PortalConstExecution( tmpInput,
                                         this->NumberOfValues,
                                         portal.Functor );
    this->PortalValid = true;
  }

  DAX_CONT_EXPORT void LoadDataForInPlace(PortalControl daxNotUsed(portal))
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output or in place.");
  }

  DAX_CONT_EXPORT void AllocateArrayForOutput(
      ContainerType &daxNotUsed(controlArray),
      dax::Id daxNotUsed(numberOfValues))
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output.");
  }
  DAX_CONT_EXPORT void RetrieveOutputData(
      ContainerType &daxNotUsed(controlArray)) const
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays cannot be used for output.");
  }

  template <class IteratorTypeControl>
  DAX_CONT_EXPORT void CopyInto(IteratorTypeControl dest) const
  {
    DAX_ASSERT_CONT(this->PortalValid);
    //synchronize the input portal to have the latest values
    // this->InputPortal.Handle.PrepareForInput();
    (void)dest;
    // std::copy(this->InputPortal.GetIteratorBegin(),
    //           this->InputPortal.GetIteratorEnd(),
    //           dest);
  }

  DAX_CONT_EXPORT void Shrink(dax::Id daxNotUsed(numberOfValues))
  {
    throw dax::cont::ErrorControlBadValue("Implicit arrays cannot be resized.");
  }

  DAX_CONT_EXPORT PortalExecution GetPortalExecution()
  {
    throw dax::cont::ErrorControlBadValue(
          "Implicit arrays are read-only.  (Get the const portal.)");
  }
  DAX_CONT_EXPORT PortalConstExecution GetPortalConstExecution() const
  {
    DAX_ASSERT_CONT(this->PortalValid);
    return this->Portal;
  }

  DAX_CONT_EXPORT void ReleaseResources() {  }

private:
  bool PortalValid;
  dax::Id NumberOfValues;
  PortalConstExecution Portal;

  PortalConstControl InputPortal;

};

}
}
} // namespace dax::cont::internal

#endif //__dax_cont_internal_ArrayHandleTransform_h
