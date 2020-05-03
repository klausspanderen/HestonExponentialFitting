/* -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*- */

/*
    Copyright (c) 2020, Klaus Spanderen
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this
    list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
    
    3. Neither the names of the copyright holders nor the names of the QuantLib   
    Group and its contributors may be used to endorse or promote products
    derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

/* The class MP_TqrEigenDecomposition is derived from 
 * QuantLib's TqrEigenDecomposition class. MPHestonVanillaEngine and HestonCvChF
 * share some code with AnalyticHestonEngine.
 * 
 * see http://quantlib.org/license.shtml
 */ 

/*! \file mphestonengine.cpp
*/

#include <ql/pricingengines/vanilla/mphestonvanillaengine.hpp>
#include <boost/math/distributions/normal.hpp>

namespace QuantLib {

    namespace {
        typedef MPHestonVanillaEngine::MP_Real MP_Real;

        const MP_Real MP_Real_MPI = 1.0/boost::math::constants::pi<MP_Real>();

        class MP_TqrEigenDecomposition {
          public:

            MP_TqrEigenDecomposition(const std::vector<MP_Real>& diag,
                                     const std::vector<MP_Real>& sub)
            : iter_(0),
              d_(diag),
              ev_(d_.size(), 0.0) {
                Size n = diag.size();

                QL_REQUIRE(n == sub.size()+1, "Wrong dimensions");

                std::vector<MP_Real> e(n, 0.0);
                std::copy(sub.begin(),sub.end(),e.begin()+1);

                ev_[0] = 1.0;

                for (Size k=n-1; k >=1; --k) {
                    while (!offDiagIsZero(k, e)) {
                        Size l = k;
                        while (--l > 0 && !offDiagIsZero(l,e));
                        iter_++;

                        MP_Real q = d_[l];
                        // calculated eigenvalue of 2x2 sub matrix of
                        // [ d_[k-1] e_[k] ]
                        // [  e_[k]  d_[k] ]
                        // which is closer to d_[k+1].
                        const MP_Real t1 = sqrt(
                                              0.25*(d_[k]*d_[k] + d_[k-1]*d_[k-1])
                                              - 0.5*d_[k-1]*d_[k] + e[k]*e[k]);
                        const MP_Real t2 = 0.5*(d_[k]+d_[k-1]);

                        const MP_Real lambda =
                            (abs(t2+t1 - d_[k]) < abs(t2-t1 - d_[k]))?
                            MP_Real(t2+t1) : MP_Real(t2-t1);

                        q-=((k==n-1)? 1.25 : 1.0)*lambda;

                        // the QR transformation
                        MP_Real sine = 1.0;
                        MP_Real cosine = 1.0;
                        MP_Real u = 0.0;

                        bool recoverUnderflow = false;
                        for (Size i=l+1; i <= k && !recoverUnderflow; ++i) {
                            const MP_Real h = cosine*e[i];
                            const MP_Real p = sine*e[i];

                            e[i-1] = sqrt(p*p+q*q);
                            if (e[i-1] != MP_Real(0.0)) {
                                sine = p/e[i-1];
                                cosine = q/e[i-1];

                                const MP_Real g = d_[i-1]-u;
                                const MP_Real t = (d_[i]-g)*sine+2*cosine*h;

                                u = sine*t;
                                d_[i-1] = g + u;
                                q = cosine*t - h;

                                const MP_Real tmp = ev_[i-1];
                                ev_[i-1] = sine*ev_[i] + cosine*tmp;
                                ev_[i] = cosine*ev_[i] - sine*tmp;
                            } else {
                                // recover from underflow
                                d_[i-1] -= u;
                                e[l] = 0.0;
                                recoverUnderflow = true;
                            }
                        }

                        if (!recoverUnderflow) {
                            d_[k] -= u;
                            e[k] = q;
                            e[l] = 0.0;
                        }
                    }
                }

                // sort (eigenvalues, eigenvectors),
                // code taken from symmetricSchureDecomposition.cpp
                std::vector<std::pair<MP_Real, MP_Real> > temp(n);
                for (Size i=0; i<n; i++) {
                    temp[i] = std::make_pair(d_[i], ev_[i]);
                }
                std::sort(temp.begin(), temp.end(),
                          std::greater<std::pair<MP_Real, MP_Real> >());
                // first element is positive
                for (Size i=0; i<n; i++) {
                    d_[i] = temp[i].first;
                    if (temp[i].second < MP_Real(0.0))
                        ev_[i] = - temp[i].second;
                    else
                        ev_[i] = temp[i].second;
                }
            }

            const std::vector<MP_Real>& eigenvalues()  const {
                return d_;
            }
            const std::vector<MP_Real>& eigenvectors() const {
                return ev_;
            }

            Size iterations() const { return iter_; }

          private:

            // see NR for abort assumption as it is
            // not part of the original Wilkinson algorithm
            bool offDiagIsZero(Size k, const std::vector<MP_Real>& e) const {
                return abs(d_[k-1])+abs(d_[k])
                    == abs(d_[k-1])+abs(d_[k])+abs(e[k]);
            }

            Size iter_;
            std::vector<MP_Real> d_;
            std::vector<MP_Real> ev_;
        };

        class MP_GausLaguerreQuadrature {
          public:
            MP_GausLaguerreQuadrature(Size n)
            : x_(n), w_(n) {
                std::vector<MP_Real> e(n-1);

                for (Size i=1; i < n; ++i) {
                    e[i-1] = MP_Real(i);
                    x_[i] = 2*e[i-1]+1;
                }
                x_[0] = MP_Real(1.0);

                MP_TqrEigenDecomposition tqr(x_, e);

                x_ = tqr.eigenvalues();
                const std::vector<MP_Real>& ev = tqr.eigenvectors();

                for (Size i=0; i<n; ++i)
                    w_[i] = ev[i]*ev[i] * exp(x_[i]);
            }

            const std::vector<MP_Real>& weights() const { return w_; }
            const std::vector<MP_Real>& x()       const { return x_; }

          protected:
            std::vector<MP_Real> x_, w_;
        };

        class HestonCvChF {
          public:
            HestonCvChF(MP_Real kappa, MP_Real sigma, MP_Real theta,
                        MP_Real rho, MP_Real v0, MP_Real t,
                        MP_Real freq, MP_Real sigmaBS)
            : kappa_(kappa),
              sigma_(sigma), sigma2_(sigma*sigma), invSigma2_(1.0/sigma2_),
              theta_(theta), rho_(rho), v0_(v0), t_(t), freq_(freq),
              sigmaBS_(sigmaBS) {}

            MP_Real operator()(MP_Real u) const {

                const std::complex<MP_Real> z(u, -0.5);
                const std::complex<MP_Real> f(0.0, u*freq_);

                const std::complex<MP_Real> phiBS =
                        std::complex<MP_Real>(-0.5*sigmaBS_*sigmaBS_*t_, 0.0)
                    *(z*z + std::complex<MP_Real>(-z.imag(), z.real()));

                return ((exp(phiBS + f)-exp(lnChF(z) + f))).real()
                        / (u*u + 0.25);
            }

          private:
            std::complex<MP_Real> lnChF(const std::complex<MP_Real>& z) const {
                const std::complex<MP_Real> g
                    (kappa_+std::complex<MP_Real>(rho_*sigma_, 0.0)
                        *std::complex<MP_Real>(z.imag(), -z.real()));

                const std::complex<MP_Real> D( sqrt(
                    g*g + (z*z +
                        std::complex<MP_Real>(-z.imag(), z.real()))*sigma2_));

                const std::complex<MP_Real> G( (g-D)/(g+D) );

                const MP_Real e1(1.0);

                return std::complex<MP_Real>(v0_*invSigma2_, 0.0)
                        *(e1-exp(-D*t_))/(e1-G*exp(-D*t_))*(g-D)
                        + std::complex<MP_Real>(kappa_*theta_*invSigma2_, 0.0)
                            *((g-D)*t_-MP_Real(2.0)*log((e1-G*exp(-D*t_))/(e1-G)));
            }

            const MP_Real kappa_, sigma_, sigma2_, invSigma2_;
            const MP_Real theta_, rho_, v0_, t_, freq_, sigmaBS_;
        };
    }

    MPHestonVanillaEngine::MPHestonVanillaEngine(
          const ext::shared_ptr<HestonModel>& model,Size n)
      : n_(n), model_(model) {
        const MP_GausLaguerreQuadrature q(n_);

        x_ = q.x();
        w_ = q.weights();
    }


    void MPHestonVanillaEngine::calculate() const {
        QL_REQUIRE(arguments_.exercise->type() == Exercise::European,
                   "not an European option");

        const Date maturityDate = arguments_.exercise->lastDate();

        const ext::shared_ptr<PlainVanillaPayoff> payoff =
            ext::dynamic_pointer_cast<PlainVanillaPayoff>(arguments_.payoff);
        QL_REQUIRE(payoff, "non plain vanilla payoff given");

        const MP_Real strike = payoff->strike();

        const ext::shared_ptr<HestonProcess> process = model_->process();

        const MP_Real t = process->time(maturityDate);

        const MP_Real rd = process->riskFreeRate()->discount(maturityDate);
        const MP_Real dd = process->dividendYield()->discount(maturityDate);

        const MP_Real spot = process->s0()->value();
        QL_REQUIRE(spot > 0.0, "negative or null underlying given");

        const MP_Real freq = log(spot) - log(rd/dd) - log(strike);

        const MP_Real fwd = spot * dd / rd;

        const MP_Real kappa = model_->kappa();
        const MP_Real sigma = model_->sigma();
        const MP_Real theta = model_->theta();
        const MP_Real rho   = model_->rho();
        const MP_Real v0    = model_->v0();

        const MP_Real vAvg = (1-exp(-kappa*t))*(v0-theta)/(kappa*t) + theta;
        const MP_Real var = vAvg*t;
        const MP_Real sqVar = sqrt(var);


        const MP_Real d1 = (log(fwd/strike) + 0.5*var)/sqVar;
        const MP_Real d2 = d1 - sqVar;

        const boost::math::normal_distribution<MP_Real> n(0.0, 1.0);
        const MP_Real bsPrice =
            rd*(fwd*boost::math::cdf(n, d1)    - strike*boost::math::cdf(n, d2));

        const HestonCvChF helper(
            kappa, sigma, theta, rho, v0, t, freq, sqrt(vAvg));

        MP_Real s = 0.0;
        for (Integer i = x_.size()-1; i >= 0; --i)
            s += w_[i] * helper(x_[i]);

        const MP_Real h_cv = s * sqrt(strike * fwd)*rd * MP_Real_MPI;

        switch (payoff->optionType())
        {
          case Option::Call:
              results_.value = (bsPrice + h_cv).convert_to<Real>();
            break;
          case Option::Put:
              results_.value = (bsPrice + h_cv - rd*(fwd - strike))
                    .convert_to<Real>();
            break;
          default:
            QL_FAIL("unknown option type");
        }
    }
}

