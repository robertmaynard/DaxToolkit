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
#ifndef __dax_cont_dispatcher_CollectCount_h
#define __dax_cont_dispatcher_CollectCount_h


#include <dax/Types.h>
#include <dax/cont/arg/ConceptMap.h>

#include <boost/mpl/if.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>

namespace dax { namespace cont { namespace dispatcher {

template <class DomainType>
class CollectCount
{
  dax::Id& Count;
public:
  CollectCount(dax::Id& c):
    Count(c)
  { this->Count = 1; }

  template <typename C, typename A>
  void operator()(const dax::cont::arg::ConceptMap<C,A>& c)
    {
    //determine the concept and traits of the concept we
    //have. This will be used to look up the domains that the concept
    //has, and if one of those domains match our domina, we know we
    //can ask it for the size/len
    typedef dax::cont::arg::ConceptMap<C,A> ConceptType;
    typedef dax::cont::arg::ConceptMapTraits<ConceptType> Traits;
    typedef typename Traits::DomainTag DomainTag;
    typedef typename boost::is_same<DomainTag,DomainType>::type HasDomain;

    this->getCount<ConceptType,HasDomain>(c);
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::enable_if<HasDomain>::type
  getCount(const ConceptType& concept)
    {
    //we can't discount all out tagged arrays, since we allow scheduling with
    //out arrays that already have been allocated.
    //what we can do is ignore out tags that have a size of zero
    typedef dax::cont::arg::ConceptMapTraits<ConceptType> Traits;
    typedef typename Traits::Tags Tags;

    typedef typename boost::mpl::if_<
        typename Tags::template Has<dax::cont::sig::Out>,
        boost::true_type, boost::false_type >::type  IgnoreZeroLength;

    dax::Id c = concept.GetDomainLength(DomainType());
    if(IgnoreZeroLength() == boost::true_type() && c == 0)
      {
      }
    else if(this->Count == 1 && c > this->Count)
      {
      //count is equal to 1 so are on the first item that is larger than
      //than the default iteration of 1. This will become the value
      //we use to evaluate all other lengths to determine the proper length
      this->Count = c;
      }
    else if(c < this->Count)
      {
      this->Count = c;
      }
    }

  template <typename ConceptType, typename HasDomain>
  typename boost::disable_if<HasDomain>::type
  getCount(const ConceptType&)
    {
      //this concept map doesn't have the domain tag we are interested
      //in so we ignore it
    }
};

} } }

#endif //__dax_cont_dispatcher_CollectCount_h
