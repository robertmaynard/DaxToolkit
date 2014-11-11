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
#ifndef __dax_exec_arg_BindCellPoints_h
#define __dax_exec_arg_BindCellPoints_h
#if defined(DAX_DOXYGEN_ONLY)

#else // !defined(DAX_DOXYGEN_ONLY)

#include <dax/Types.h>
#include <dax/CellTag.h>
#include <dax/VectorTraits.h>

#include <dax/cont/arg/ConceptMap.h>
#include <dax/cont/arg/Topology.h>

#include <dax/exec/arg/ArgBase.h>
#include <dax/exec/arg/BindInfo.h>
#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>
#include <dax/exec/internal/IJKIndex.h>
#include <dax/exec/internal/WorkletBase.h>

#include <boost/utility/enable_if.hpp>

namespace dax { namespace exec { namespace arg {

template <typename Invocation, int N>
class BindCellPoints
    : public dax::exec::arg::ArgBase<BindCellPoints<Invocation,N> >
{
  typedef dax::exec::arg::ArgBaseTraits<
      BindCellPoints<Invocation, N > > Traits;

  enum{TopoIndex=Traits::TopoIndex};
  typedef typename Traits::TopoExecArgType TopoExecArgType;
  typedef typename Traits::ExecArgType ExecArgType;
  typedef typename TopoExecArgType::CellTag CellTag;

public:

  typedef typename Traits::ValueType ValueType;
  typedef typename Traits::ReturnType ReturnType;
  typedef typename Traits::SaveType SaveType;

  DAX_CONT_EXPORT BindCellPoints(
      typename dax::cont::internal::Bindings<Invocation>::type &bindings):
    TopoExecArg(dax::exec::arg::GetNthExecArg<TopoIndex>(bindings)),
    ExecArg(dax::exec::arg::GetNthExecArg<N>(bindings)),
    Value(typename dax::VectorTraits<ValueType>::ComponentType())
    {
      //now that we have the ExecArg and TopoExecArgs, we should tell them
      //that the access patterns will be random access, this way if
      //the backend can map it to texture memory etc. Hmm this
      //might have to happen at ArgBaseTraits construction time

      //How does the FieldArrayHandle know that we are a cell worklet
      //and are binding a point field. Can The dispatcher help out here?
      //If we can just tell FieldArrayHandle that it is going to be
      //used for random access it would use a const random access portal
      //maybe something like:
      /*

      typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

      typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Point>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasPointTag;

      typedef typename ::boost::mpl::and_< typename HasPointTag::type,
                                           typename HasInTag::type >::type HasInPointTag;

      typedef typename boost::mpl::if_<
              typename Tags::template Has<dax::cont::sig::Out>,
              typename HandleType::PortalExecution,
              typename HandleType::PortalConstExecution>::type  FallBackPortalType;

      typedef typename boost::mpl::if_< typename HasInPointTag::type
                       typename HandleType::PortalConstRandomAccessExecution,
                       FallBackPortalType>::type PortalType;


      We have the Device adapter inside the FieldHandle so we can specialize
      the array handle
      */

    }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForWriting(const IndexType&,
                            const dax::exec::internal::WorkletBase&)
    { return this->Value; }

  template<typename IndexType>
  DAX_EXEC_EXPORT ReturnType GetValueForReading(
                            const IndexType& index,
                            const dax::exec::internal::WorkletBase& work) const
    {
    ValueType v;
    const dax::exec::CellVertices<CellTag>& pointIndices =
                                            this->TopoExecArg(index, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      v[vertexIndex] = this->ExecArg(pointIndices[vertexIndex],work);
      }
    return v;
    }

  DAX_EXEC_EXPORT void SaveValue(int index,
                        const dax::exec::internal::WorkletBase& work) const
    {
    this->SaveValue(index,this->Value,work);
    }

  DAX_EXEC_EXPORT void SaveValue(int index, const SaveType& v,
                        const dax::exec::internal::WorkletBase& work) const
    {
    const dax::exec::CellVertices<CellTag>& pointIndices =
                                            this->TopoExecArg(index, work);
    for(int vertexIndex = 0;
        vertexIndex < pointIndices.NUM_VERTICES;
        ++vertexIndex)
      {
      this->ExecArg.SaveExecutionResult(pointIndices[vertexIndex],
                                        v[vertexIndex],
                                        work);
      }
    }
private:
  TopoExecArgType TopoExecArg;
  ExecArgType ExecArg;
  ValueType Value;
};



//the traits for BindPermutedCellField
template <typename Invocation, int N >
struct ArgBaseTraits< BindCellPoints<Invocation, N> >
{
private:
  typedef dax::exec::arg::FindBindInfo<
      dax::cont::arg::Topology,Invocation> TopoInfo;
  typedef dax::exec::arg::BindInfo<N,Invocation> MyInfo;
  typedef typename MyInfo::Tags Tags;
public:
  enum{TopoIndex=TopoInfo::Index};

  typedef typename TopoInfo::ExecArgType TopoExecArgType;
  typedef typename MyInfo::ExecArgType ExecArgType;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::Out>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasOutTag;

  typedef typename ::boost::mpl::if_<typename Tags::template Has<dax::cont::sig::In>,
                                   ::boost::true_type,
                                   ::boost::false_type>::type HasInTag;

  typedef dax::exec::CellField<typename ExecArgType::ValueType,
                               typename TopoExecArgType::CellTag> ValueType;

  typedef typename boost::mpl::if_<typename HasOutTag::type,
                                   ValueType&,
                                   ValueType const>::type ReturnType;
  typedef ValueType SaveType;
};

}}} // namespace dax::exec::arg

#endif // !defined(DAX_DOXYGEN_ONLY)
#endif //__dax_exec_arg_BindCellPoints_h
