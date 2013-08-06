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

#ifndef __MarchingCubes_worklet_
#define __MarchingCubes_worklet_

#include <dax/CellTag.h>
#include <dax/CellTraits.h>
#include <dax/exec/CellField.h>
#include <dax/exec/CellVertices.h>

#include <dax/exec/WorkletGenerateKeysValues.h>
#include <dax/exec/WorkletMapCell.h>
#include <dax/exec/WorkletReduceKeysValues.h>
#include <dax/exec/WorkletGenerateTopology.h>

#include <dax/worklet/internal/MarchingCubesTable.h>

namespace dax {
namespace worklet {

namespace internal{
namespace marchingcubes{
// -----------------------------------------------------------------------------
template<typename T, typename U>
DAX_EXEC_EXPORT
int GetHexahedronClassification(const T isoValue, const U& values )
{
  return ((values[0] > isoValue) << 0 |
          (values[1] > isoValue) << 1 |
          (values[2] > isoValue) << 2 |
          (values[3] > isoValue) << 3 |
          (values[4] > isoValue) << 4 |
          (values[5] > isoValue) << 5 |
          (values[6] > isoValue) << 6 |
          (values[7] > isoValue) << 7);
}
}
}

// -----------------------------------------------------------------------------
class MarchingCubesClassify : public dax::exec::WorkletMapCell
{
public:
  typedef void ControlSignature(Topology, Field(Point), Field(Out));
  typedef _3 ExecutionSignature(_2);

  DAX_CONT_EXPORT MarchingCubesClassify(dax::Scalar isoValue)
    : IsoValue(isoValue) {  }

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id operator()(
      const dax::exec::CellField<dax::Scalar,CellTag> &values) const
  {
    // If you get a compile error on the following line, it means that this
    // worklet was used with an improper cell type.  Check the cell type for the
    // input grid given in the control environment.
    return this->GetNumFaces(
          values,
          typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }
private:
  dax::Scalar IsoValue;

  template<class CellTag>
  DAX_EXEC_EXPORT
  dax::Id GetNumFaces(const dax::exec::CellField<dax::Scalar,CellTag> &values,
                      dax::CellTagHexahedron) const
  {
    const int voxelClass =
    internal::marchingcubes::GetHexahedronClassification(IsoValue,values);
    return 3 * dax::worklet::internal::marchingcubes::NumFaces[voxelClass];
  }
};

// -----------------------------------------------------------------------------
class MarchingCubesGenerate : public dax::exec::WorkletGenerateKeysValues
{
public:

  typedef void ControlSignature(Topology, Field(Point,In), Field(Out), Field(Out)) ;
  typedef void ExecutionSignature(Vertices(_1), _2, _3, _4, VisitIndex);

  DAX_CONT_EXPORT MarchingCubesGenerate(dax::Scalar isoValue)
    : IsoValue(isoValue){ }

  template<class CellTag>
  DAX_EXEC_EXPORT void operator()(
      const dax::exec::CellVertices<CellTag>& verts,
      const dax::exec::CellField<dax::Scalar,CellTag> &values,
      dax::Id2& outKey,
      dax::Scalar& outValue,
      dax::Id inputCellVisitIndex) const
  {
    // If you get a compile error on the following line, it means that this
    // worklet was used with an improper cell type.  Check the cell type for the
    // input grid given in the control environment.
    this->BuildKeyValue(
          verts,
          values,
          outKey,
          outValue,
          inputCellVisitIndex,
          typename dax::CellTraits<CellTag>::CanonicalCellTag());
  }

private:
  dax::Scalar IsoValue;

  template<class CellTag>
  DAX_EXEC_EXPORT void BuildKeyValue(
      const dax::exec::CellVertices<CellTag>& verts,
      const dax::exec::CellField<dax::Scalar,CellTag> &values,
      dax::Id2& outKey,
      dax::Scalar& outValue,
      dax::Id inputCellVisitIndex,
      dax::CellTagHexahedron) const
  {
    using dax::worklet::internal::marchingcubes::TriTable;
    // These should probably be available through the voxel class
    const unsigned char voxelVertEdges[12][2] ={
        {0,1}, {1,2}, {3,2}, {0,3},
        {4,5}, {5,6}, {7,6}, {4,7},
        {0,4}, {1,5}, {2,6}, {3,7},
      };

    const int voxelClass =
        internal::marchingcubes::GetHexahedronClassification(IsoValue,values);

    //save the point ids and ratio to interpolate the points of the new cell
    const unsigned char edge = TriTable[voxelClass][inputCellVisitIndex];
    const int vertA = voxelVertEdges[edge][0];
    const int vertB = voxelVertEdges[edge][1];

    // Find the weight for linear interpolation
    const dax::Scalar weight = (IsoValue - values[vertA]) /
                               (values[vertB]-values[vertA]);

    outKey = dax::Id2(verts[vertA],verts[vertB]);
    outValue = weight;
  }
};

// -----------------------------------------------------------------------------
class MarchingCubesInterpolate : public dax::exec::WorkletReduceKeysValues
{
public:
  typedef void ControlSignature(Values(In), Values(Out));
  typedef void ExecutionSignature( KeyGroup(_1), _2);

  template<typename KeyGroupType, typename FieldType>
  DAX_EXEC_EXPORT
  void operator()(KeyGroupType inPortal, FieldType &reducedValue) const
  {
  reducedValue = inPortal.Get(0);
  }
};


}
} //dax::worklet

#endif
