# References

Keys cited by docstrings in the code and by [`FORMULAS.md`](FORMULAS.md), in the form
`[KEY, eq. n]` or `[KEY, ch. n]`. Add a key here before citing it; keep the list alphabetical.

| key | reference |
|---|---|
| [AE04] | A. Abur and A. Gómez Expósito, *Power System State Estimation: Theory and Implementation*, Marcel Dekker, 2004. |
| [ASP14] | M. Asprou, E. Kyriakides and M. Albu, "The effect of variable weights in a WLS state estimator considering instrument transformer uncertainties," IEEE Trans. Instrum. Meas., vol. 63, no. 6, pp. 1484–1495, 2014, and the companion "The effect of instrument transformer accuracy class on the WLS state estimator accuracy," IEEE PES General Meeting, 2013. The accuracy-class meter error model (maximum error divided by √3). |
| [DAT26] | The fdia-graph dataset description: this repository's README and `docs/reference/DATA_DICTIONARY.md` (attack families, plausibility band, ramp profile). |
| [FED26] | Our federated per-bus FDIA localization work (the swing and temporal-delta features); see `docs/localization/README.md`. |
| [HAN75] | E. Handschin, F. C. Schweppe, J. Kohlas and A. Fiechter, "Bad data analysis for power system state estimation," IEEE Trans. Power App. Syst., vol. PAS-94, no. 2, 1975. Normalized residuals and the residual covariance. |
| [HUB64] | P. J. Huber, "Robust estimation of a location parameter," Ann. Math. Statist., vol. 35, no. 1, 1964. The Huber weight function. |
| [JAC26] | R. Abdulin and R. Narimani, Jacobian-informed features for FDIA localization (working digest, 2026); see `docs/localization/README.md`, section "Jacobian-informed features". |
| [MP19] | R. D. Zimmerman and C. E. Murillo-Sánchez, *MATPOWER User's Manual*, version 7.0, 2019. Branch model and `makeYbus`. |
| [SCH70] | F. C. Schweppe and J. Wildes, "Power system static-state estimation, Part I: Exact model"; F. C. Schweppe and D. B. Rom, "Part II: Approximate model"; F. C. Schweppe, "Part III: Implementation," IEEE Trans. Power App. Syst., vol. PAS-89, no. 1, 1970. |

Entries marked "our" or "working digest" are the group's own work and are cited so the code
points at the document that defines the quantity, not at a published paper.
