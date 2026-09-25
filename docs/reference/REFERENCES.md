# References

Keys cited by docstrings in the code and by [`FORMULAS.md`](FORMULAS.md), in the form
`[KEY, eq. n]` or `[KEY, ch. n]`. Add a key here before citing it; keep the list alphabetical.

| key | reference |
|---|---|
| [AE04] | A. Abur and A. Gómez Expósito, *Power System State Estimation: Theory and Implementation*, Marcel Dekker, 2004. |
| [ASP14] | M. Asprou, E. Kyriakides and M. Albu, "The effect of variable weights in a WLS state estimator considering instrument transformer uncertainties," IEEE Trans. Instrum. Meas., vol. 63, no. 6, pp. 1484–1495, 2014, and the companion "The effect of instrument transformer accuracy class on the WLS state estimator accuracy," IEEE PES General Meeting, 2013. The accuracy-class meter error model (maximum error divided by √3). |
| [BOY22] | O. Boyaci, A. Umunnakwe, A. Sahu, M. R. Narimani, M. Ismail, K. R. Davis and E. Serpedin, "Graph neural networks based detection of stealth false data injection attacks in smart grids," IEEE Systems Journal, vol. 16, no. 2, pp. 2946–2957, 2022. Algorithm 1: every load and generator at a bus scaled by one factor per time step; the local attacker skips generator and zero-injection buses. |
| [CGL79] | T. F. Chan, G. H. Golub and R. J. LeVeque, "Updating formulae and a pairwise algorithm for computing sample variances," Stanford University, Tech. Rep. STAN-CS-79-773, 1979. Pooling means and variances of disjoint parts. |
| [DAT26] | The fdia-graph dataset description: this repository's README and `docs/reference/DATA_DICTIONARY.md` (attack families, plausibility band, ramp profile). |
| [DG06] | J. Davis and M. Goadrich, "The relationship between precision-recall and ROC curves," Proc. 23rd Int. Conf. Machine Learning (ICML), pp. 233–240, 2006. Average precision as the step area under the precision-recall curve. |
| [EST26] | The project's state-estimation work (the subspace prior, its Huber composition and the localization gate); see `docs/se/README.md`. |
| [FED26] | The project's federated per-bus FDIA localization work (the swing and temporal-delta features); see `docs/localization/README.md`. |
| [HAG84] | W. W. Hager, "Condition estimates," SIAM J. Sci. Stat. Comput., vol. 5, no. 2, pp. 311–316, 1984. The 1-norm estimate of an inverse from a few solves, the fallback when SciPy lacks LAPACK's dtrcon. |
| [HAN75] | E. Handschin, F. C. Schweppe, J. Kohlas and A. Fiechter, "Bad data analysis for power system state estimation," IEEE Trans. Power App. Syst., vol. PAS-94, no. 2, 1975. Normalized residuals and the residual covariance. |
| [HUB64] | P. J. Huber, "Robust estimation of a location parameter," Ann. Math. Statist., vol. 35, no. 1, 1964. The Huber weight function. |
| [JAC26] | R. Abdulin and R. Narimani, Jacobian-informed features for FDIA localization (working digest, 2026); see `docs/localization/README.md`, section "Jacobian-informed features". |
| [KEC25] | C. Keçeci, K. R. Davis and E. Serpedin, "Federated learning-based distributed localization of false data injection attacks on smart grids," IEEE Systems Journal, vol. 19, no. 3, pp. 719–729, 2025. Per-bus localization metrics. |
| [MCM17] | B. McMahan, E. Moore, D. Ramage, S. Hampson and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," Proc. 20th Int. Conf. Artificial Intelligence and Statistics (AISTATS), pp. 1273–1282, 2017. Federated averaging. |
| [MP19] | R. D. Zimmerman and C. E. Murillo-Sánchez, *MATPOWER User's Manual*, version 7.0, 2019. Branch model and `makeYbus`. |
| [SCH70] | F. C. Schweppe and J. Wildes, "Power system static-state estimation, Part I: Exact model"; F. C. Schweppe and D. B. Rom, "Part II: Approximate model"; F. C. Schweppe, "Part III: Implementation," IEEE Trans. Power App. Syst., vol. PAS-89, no. 1, 1970. |
| [VLX07] | U. von Luxburg, "A tutorial on spectral clustering," Statistics and Computing, vol. 17, no. 4, pp. 395–416, 2007. The spectral partition of the buses into clients. |
| [WU26] | Y. Wu, H. Wang, S. Hu, J. Ye and Y. Tang, "Dynamic PMU configuration for stealthy multi-snapshot FDIA mitigation," IEEE Trans. Smart Grid, vol. 17, no. 1, pp. 650–665, 2026. The multi-snapshot attack: a held target reached in per-snapshot steps under the noise floor with an l0-sparse tamper set, every snapshot AC-consistent. |

[DAT26], [EST26], [FED26] and [JAC26] are the project's own work. They are cited so the code points
at the document that defines the quantity, not at a published paper.
