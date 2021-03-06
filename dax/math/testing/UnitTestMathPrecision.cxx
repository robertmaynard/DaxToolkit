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

// This teases out a bug where math functions in the std namespace conflict
// with math functions in other namespaces.
namespace std { } // make sure std defined
using namespace std;

#include <dax/math/Precision.h>

#include <dax/exec/VectorOperations.h>

#include <dax/testing/Testing.h>

namespace {

void TestNonFinites()
{
  std::cout << "Testing non-finites." << std::endl;

  dax::Scalar zero = 0.0;
  dax::Scalar finite = 1.0;
  dax::Scalar nan = dax::math::Nan();
  dax::Scalar inf = dax::math::Infinity();
  dax::Scalar neginf = dax::math::NegativeInfinity();
  dax::Scalar epsilon = dax::math::Epsilon();

  // General behavior.
  DAX_TEST_ASSERT(nan != nan, "Nan not equal itself.");
  DAX_TEST_ASSERT(!(nan >= zero), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan <= zero), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan >= finite), "Nan not greater or less.");
  DAX_TEST_ASSERT(!(nan <= finite), "Nan not greater or less.");

  DAX_TEST_ASSERT(neginf < inf, "Infinity big");
  DAX_TEST_ASSERT(zero < inf, "Infinity big");
  DAX_TEST_ASSERT(finite < inf, "Infinity big");
  DAX_TEST_ASSERT(zero > -inf, "-Infinity small");
  DAX_TEST_ASSERT(finite > -inf, "-Infinity small");
  DAX_TEST_ASSERT(zero > neginf, "-Infinity small");
  DAX_TEST_ASSERT(finite > neginf, "-Infinity small");

  DAX_TEST_ASSERT(zero < epsilon, "Negative epsilon");
  DAX_TEST_ASSERT(finite > epsilon, "Large epsilon");

  // Math check functions.
  DAX_TEST_ASSERT(!dax::math::IsNan(zero), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::math::IsNan(finite), "Bad IsNan check.");
  DAX_TEST_ASSERT(dax::math::IsNan(nan), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::math::IsNan(inf), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::math::IsNan(neginf), "Bad IsNan check.");
  DAX_TEST_ASSERT(!dax::math::IsNan(epsilon), "Bad IsNan check.");

  DAX_TEST_ASSERT(!dax::math::IsInf(zero), "Bad infinity check.");
  DAX_TEST_ASSERT(!dax::math::IsInf(finite), "Bad infinity check.");
  DAX_TEST_ASSERT(!dax::math::IsInf(nan), "Bad infinity check.");
  DAX_TEST_ASSERT(dax::math::IsInf(inf), "Bad infinity check.");
  DAX_TEST_ASSERT(dax::math::IsInf(neginf), "Bad infinity check.");
  DAX_TEST_ASSERT(!dax::math::IsInf(epsilon), "Bad infinity check.");

  DAX_TEST_ASSERT(dax::math::IsFinite(zero), "Bad finite check.");
  DAX_TEST_ASSERT(dax::math::IsFinite(finite), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::math::IsFinite(nan), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::math::IsFinite(inf), "Bad finite check.");
  DAX_TEST_ASSERT(!dax::math::IsFinite(neginf), "Bad finite check.");
  DAX_TEST_ASSERT(dax::math::IsFinite(epsilon), "Bad finite check.");
}

template<typename VectorType>
void TestFMod(VectorType numerator,
              VectorType denominator,
              VectorType remainder)
{
  std::cout << "  Testing FMod "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computed = dax::math::FMod(numerator, denominator);
  DAX_TEST_ASSERT(test_equal(computed, remainder), "Bad remainder");
}

template<typename VectorType>
void TestRemainder(VectorType numerator,
                   VectorType denominator,
                   VectorType remainder,
                   VectorType quotient)
{
  std::cout << "  Testing Remainder "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computedRemainder
      = dax::math::Remainder(numerator, denominator);
  DAX_TEST_ASSERT(test_equal(computedRemainder, remainder), "Bad remainder");

  dax::exec::VectorFill(computedRemainder, dax::Scalar(0.0));
  VectorType computedQuotient;

  computedRemainder = dax::math::RemainderQuotient(numerator,
                                                   denominator,
                                                   computedQuotient);
  DAX_TEST_ASSERT(test_equal(computedRemainder, remainder), "Bad remainder");
  DAX_TEST_ASSERT(test_equal(computedQuotient, quotient), "Bad quotient");

  typedef dax::VectorTraits<VectorType> Traits;
  int iQuotient;
  dax::Scalar sRemainder;
  sRemainder
      = dax::math::RemainderQuotient(Traits::GetComponent(numerator, 0),
                                     Traits::GetComponent(denominator, 0),
                                     iQuotient);
  DAX_TEST_ASSERT(test_equal(sRemainder, Traits::GetComponent(remainder, 0)),
                  "Bad remainder");
  DAX_TEST_ASSERT(test_equal(dax::Scalar(iQuotient),
                             Traits::GetComponent(quotient, 0)),
                  "Bad quotient");
}

template<typename VectorType>
void TestModF(VectorType x, VectorType integral, VectorType fractional)
{
  std::cout << "  Testing ModF "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  VectorType computedIntegral;
  VectorType computedFractional;

  computedFractional = dax::math::ModF(x, computedIntegral);

  DAX_TEST_ASSERT(test_equal(computedIntegral, integral), "Bad integral");
  DAX_TEST_ASSERT(test_equal(computedFractional, fractional), "Bad fractional");
}

template<typename VectorType>
void TestRound(VectorType x,
               VectorType xFloor,
               VectorType xCeil,
               VectorType xRound)
{
  std::cout << "  Testing Round "
            << dax::VectorTraits<VectorType>::NUM_COMPONENTS << " components"
            << std::endl;

  DAX_TEST_ASSERT(test_equal(dax::math::Floor(x), xFloor), "Bad floor");
  DAX_TEST_ASSERT(test_equal(dax::math::Ceil(x), xCeil), "Bad ceil");
  DAX_TEST_ASSERT(test_equal(dax::math::Round(x), xRound), "Bad round");
}

const dax::Id MAX_VECTOR_SIZE = 4;

const dax::Scalar NumeratorInit[MAX_VECTOR_SIZE] =   { 6.5, 5.8, 9.3, 77.0 };
const dax::Scalar DenominatorInit[MAX_VECTOR_SIZE] = { 2.3, 1.6, 3.1, 19.0 };
const dax::Scalar FModRemainderInit[MAX_VECTOR_SIZE]={ 1.9, 1.0, 0.0,  1.0 };
const dax::Scalar RemainderInit[MAX_VECTOR_SIZE] =   {-0.4,-0.6, 0.0,  1.0 };
const dax::Scalar QuotientInit[MAX_VECTOR_SIZE] =    { 3.0, 4.0, 3.0,  4.0 };

const dax::Scalar XInit[MAX_VECTOR_SIZE] =           {4.6, 0.1, 73.4, 55.0 };
const dax::Scalar FractionalInit[MAX_VECTOR_SIZE] =  {0.6, 0.1,  0.4,  0.0 };
const dax::Scalar FloorInit[MAX_VECTOR_SIZE] =       {4.0, 0.0, 73.0, 55.0 };
const dax::Scalar CeilInit[MAX_VECTOR_SIZE] =        {5.0, 1.0, 74.0, 55.0 };
const dax::Scalar RoundInit[MAX_VECTOR_SIZE] =       {5.0, 0.0, 73.0, 55.0 };

struct TestPrecisionFunctor
{
  template <typename T> void operator()(const T&) const {
    typedef dax::VectorTraits<T> Traits;
    DAX_TEST_ASSERT(Traits::NUM_COMPONENTS <= MAX_VECTOR_SIZE,
                    "Need to update test for larger vectors.");
    T numerator, denominator, fmodremainder, remainder, quotient;
    T x, fractional, floor, ceil, round;
    for (int index = 0; index < Traits::NUM_COMPONENTS; index++)
      {
      Traits::SetComponent(numerator, index, NumeratorInit[index]);
      Traits::SetComponent(denominator, index, DenominatorInit[index]);
      Traits::SetComponent(fmodremainder, index, FModRemainderInit[index]);
      Traits::SetComponent(remainder, index, RemainderInit[index]);
      Traits::SetComponent(quotient, index, QuotientInit[index]);

      Traits::SetComponent(x, index, XInit[index]);
      Traits::SetComponent(fractional, index, FractionalInit[index]);
      Traits::SetComponent(floor, index, FloorInit[index]);
      Traits::SetComponent(ceil, index, CeilInit[index]);
      Traits::SetComponent(round, index, RoundInit[index]);
      }
    TestFMod(numerator, denominator, fmodremainder);
    TestRemainder(numerator, denominator, remainder, quotient);
    TestModF(x, floor, fractional);
    TestRound(x, floor, ceil, round);
  }
};

void TestPrecision()
{
  TestNonFinites();
  dax::testing::Testing::TryAllTypes(TestPrecisionFunctor(),
                                      dax::testing::Testing::TypeCheckReal());
}

} // anonymous namespace

int UnitTestMathPrecision(int, char *[])
{
  return dax::testing::Testing::Run(TestPrecision);
}
