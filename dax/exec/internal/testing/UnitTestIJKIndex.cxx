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

#include <dax/exec/internal/GridTopologies.h>

#include <dax/Extent.h>
#include <dax/CellTag.h>
#include <dax/exec/internal/IJKIndex.h>
#include <dax/testing/Testing.h>
#include <vector>

namespace {

using dax::exec::internal::IJKIndex;

template<typename Grid>
static void TestGridIJK(const Grid &gridstruct)
{

  dax::Id3 dims = dax::extentCellDimensions(gridstruct.Extent);
  IJKIndex ijkIndex(dims);
  dax::Id correctFlatCellIndex=0;
  for(dax::Id k = 0; k < dims[2]; ++k)
    {
    ijkIndex.SetK(k);
    for(dax::Id j = 0; j < dims[1]; ++j)
      {
      ijkIndex.SetJ(j);
      for(dax::Id i = 0; i < dims[0]; ++i)
        {
        ijkIndex.SetI(i);

        //verify the int conversion
        const IJKIndex greaterByOne(dims,dax::Id3(i,j,k+1));
        const IJKIndex lesserByOne(dims,dax::Id3(i,j,k-1));
        const IJKIndex sameIndex = ijkIndex;
        const dax::Id convertedIndex = ijkIndex;

        DAX_TEST_ASSERT(convertedIndex == correctFlatCellIndex,
                  "ijk dax::Id casting returned wrong cell index");

        //verify the == operator
        DAX_TEST_ASSERT(ijkIndex == correctFlatCellIndex,
                  "ijk operator == is wrong");
        DAX_TEST_ASSERT(ijkIndex == IJKIndex(dims,dax::Id3(i,j,k)),
          "ijk operator == is wrong");

        //verify the != operator
        DAX_TEST_ASSERT(!(ijkIndex != correctFlatCellIndex),
                "ijk != operator is wrong");
        DAX_TEST_ASSERT(!(ijkIndex != sameIndex),
          "ijk operator == is wrong");

        //verify the comparison operators
        DAX_TEST_ASSERT( (ijkIndex < (convertedIndex+1) &&
                         !(ijkIndex < convertedIndex-1)),
                        "ijk < operator is wrong");
        DAX_TEST_ASSERT( ( (!(ijkIndex >= convertedIndex+1)) &&
                           (ijkIndex >= convertedIndex-1) &&
                           (ijkIndex >= convertedIndex)),
                        "ijk >= operator is wrong");

        //verify the ijk comparison
        DAX_TEST_ASSERT( (ijkIndex < greaterByOne &&
                         !(ijkIndex < lesserByOne)),
                        "ijk < operator is wrong");
        DAX_TEST_ASSERT( (!(ijkIndex >= greaterByOne) &&
                          (ijkIndex >= lesserByOne) &&
                          (ijkIndex >= sameIndex)),
                        "ijk >= operator is wrong");

        //verify we can convert and multiply with the implicit dax::id convert
        DAX_TEST_ASSERT( (ijkIndex * 4) ==  (convertedIndex * 4),
                        "ijk * operator is wrong");


        //verify the get method returns the expect value
        const dax::Id3 value = ijkIndex.GetIJK();
        DAX_TEST_ASSERT( i == value[0] &&
                         j == value[1] &&
                         k == value[2],
                       "ijk.GetIJK returned wrong ijk values");

        ++correctFlatCellIndex;
        }
      }
    }
}

static void TestUniformGrid()
{
  std::cout << "Testing Structured grid ijk indexing." << std::endl;

  dax::exec::internal::TopologyUniform gridstruct;
  gridstruct.Origin = dax::make_Vector3(0.0, 0.0, 0.0);
  gridstruct.Spacing = dax::make_Vector3(1.0, 1.0, 1.0);

  gridstruct.Extent.Min = dax::make_Id3(0, 0, 0);
  gridstruct.Extent.Max = dax::make_Id3(10, 10, 10);
  TestGridIJK(gridstruct);

  gridstruct.Extent.Min = dax::make_Id3(-20, 10, 3);
  gridstruct.Extent.Max = dax::make_Id3(15, 25, 13);
  TestGridIJK(gridstruct);
}

}

int UnitTestIJKIndex(int, char *[])
{
  return dax::testing::Testing::Run(TestUniformGrid);
}
