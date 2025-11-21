/**
 * NCGD: Non-Commutative Geometric Dynamics
 * Core Module: SE(3) Lie Algebra - The Complete Hardcore Implementation.
 * * Features:
 * 1. Exact Exponential Map (se(3) -> SE(3)) via Rodrigues' Formula.
 * 2. Lie Bracket ([xi, xi]) for non-commutative curvature.
 * 3. Adjoint Map (Ad_T) for coordinate frame transformation.
 * * This guarantees GEOMETRIC RIGIDITY at machine precision.
 */

#ifndef NCGD_SE3_ALGEBRA_HPP
#define NCGD_SE3_ALGEBRA_HPP

#include <array>
#include <cmath>
#include <iostream>
#include <limits>

namespace ncgd {

    using Scalar = double;
    using Vector3 = std::array<Scalar, 3>;
    using Vector6 = std::array<Scalar, 6>; // Tangent Vector (Twist)
    using Matrix3x3 = std::array<std::array<Scalar, 3>, 3>;
    using Matrix4x4 = std::array<std::array<Scalar, 4>, 4>;
    using Matrix6x6 = std::array<std::array<Scalar, 6>, 6>;

    class SE3Algebra {
    private:
        static constexpr Scalar EPSILON = 1e-10;

    public:
        // ---------------------------------------------------------
        // 1. Lie Bracket [u, v] = ad_u(v)
        // Captures the non-commutativity of rigid body motions.
        // ---------------------------------------------------------
        static Vector6 lie_bracket(const Vector6& u, const Vector6& v) {
            Vector3 w_u = {u[0], u[1], u[2]}; Vector3 v_u = {u[3], u[4], u[5]};
            Vector3 w_v = {v[0], v[1], v[2]}; Vector3 v_v = {v[3], v[4], v[5]};

            Vector3 w_new = cross(w_u, w_v);
            Vector3 v_new_1 = cross(w_u, v_v);
            Vector3 v_new_2 = cross(w_v, v_u); // Note: w_v x v_u

            return {
                w_new[0], w_new[1], w_new[2],
                v_new_1[0] - v_new_2[0], 
                v_new_1[1] - v_new_2[1], 
                v_new_1[2] - v_new_2[2]
            };
        }

        // ---------------------------------------------------------
        // 2. Exponential Map: se(3) -> SE(3)
        // "The Integrator". Moves straight along the geodesic.
        // Input: Twist xi (6D). Output: Transformation Matrix T (4x4).
        // ---------------------------------------------------------
        static Matrix4x4 exp(const Vector6& xi) {
            Vector3 w = {xi[0], xi[1], xi[2]};
            Vector3 v = {xi[3], xi[4], xi[5]};
            Scalar theta_sq = w[0]*w[0] + w[1]*w[1] + w[2]*w[2];
            Scalar theta = std::sqrt(theta_sq);

            Matrix3x3 Omega = hat3(w);
            Matrix3x3 OmegaSq = mat_mul3(Omega, Omega);
            Matrix3x3 R;
            Matrix3x3 V; // Left Jacobian for translation

            // Handling singularity at theta -> 0 (Taylor Expansion)
            if (theta < EPSILON) {
                // R ~ I + [w]
                // V ~ I + 0.5 * [w]
                R = identity3();
                V = identity3();
                for(int i=0; i<3; ++i)
                    for(int j=0; j<3; ++j) {
                        R[i][j] += Omega[i][j];
                        V[i][j] += 0.5 * Omega[i][j];
                    }
            } else {
                // Exact Rodrigues' Formula
                // R = I + (sin t / t) * K + ((1 - cos t) / t^2) * K^2
                Scalar A = std::sin(theta) / theta;
                Scalar B = (1.0 - std::cos(theta)) / theta_sq;
                Scalar C = (1.0 - A) / theta_sq; // For V matrix

                Matrix3x3 I = identity3();
                for(int i=0; i<3; ++i) {
                    for(int j=0; j<3; ++j) {
                        R[i][j] = I[i][j] + A * Omega[i][j] + B * OmegaSq[i][j];
                        V[i][j] = I[i][j] + B * Omega[i][j] + C * OmegaSq[i][j];
                    }
                }
            }

            // Compute Translation t = V * v
            Vector3 t = mat_vec_mul3(V, v);

            // Assemble 4x4 SE(3) Matrix
            Matrix4x4 T;
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) T[i][j] = R[i][j];
                T[i][3] = t[i];
            }
            T[3][0]=0; T[3][1]=0; T[3][2]=0; T[3][3]=1;
            return T;
        }

        // ---------------------------------------------------------
        // 3. Adjoint Map: Ad_T
        // Transforms a twist from Body frame to Spatial frame.
        // Ad_T = [ R, 0 ]
        //        [ [t]x R, R ]
        // ---------------------------------------------------------
        static Matrix6x6 adjoint(const Matrix4x4& T) {
            Matrix3x3 R;
            Vector3 t;
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) R[i][j] = T[i][j];
                t[i] = T[i][3];
            }

            Matrix3x3 t_skew = hat3(t);
            Matrix3x3 tR = mat_mul3(t_skew, R);

            Matrix6x6 Ad;
            for(int i=0; i<6; ++i) for(int j=0; j<6; ++j) Ad[i][j] = 0.0;

            // Fill blocks
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    Ad[i][j] = R[i][j];       // Top-Left: R
                    Ad[i+3][j+3] = R[i][j];   // Bottom-Right: R
                    Ad[i+3][j] = tR[i][j];    // Bottom-Left: [t]x R
                }
            }
            return Ad;
        }

    private:
        // --- Low-level Algebra Helpers (No external deps) ---
        
        static Matrix3x3 identity3() {
            return {{{1,0,0}, {0,1,0}, {0,0,1}}};
        }

        static Matrix3x3 hat3(const Vector3& v) {
            return {{ {0, -v[2], v[1]}, {v[2], 0, -v[0]}, {-v[1], v[0], 0} }};
        }

        static Vector3 cross(const Vector3& a, const Vector3& b) {
            return {a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
        }

        static Matrix3x3 mat_mul3(const Matrix3x3& A, const Matrix3x3& B) {
            Matrix3x3 C = {};
            for(int i=0; i<3; ++i)
                for(int j=0; j<3; ++j)
                    for(int k=0; k<3; ++k)
                        C[i][j] += A[i][k] * B[k][j];
            return C;
        }

        static Vector3 mat_vec_mul3(const Matrix3x3& M, const Vector3& v) {
            Vector3 res = {0,0,0};
            for(int i=0; i<3; ++i)
                for(int j=0; j<3; ++j)
                    res[i] += M[i][j] * v[j];
            return res;
        }
    };
}

#endif // NCGD_SE3_ALGEBRA_HPP
