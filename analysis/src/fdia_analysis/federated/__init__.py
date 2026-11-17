"""Federated training over K clients (utilities) that never pool their records [FED26].

Built from the federated localization paper's harness: the partition of buses into clients
(`spectral_partition`) and the paper's localizers trained by FedAvg (`FedBusMLP`, `FedBusCNN`,
drop-in LocalizerBase methods), and the estimator's subspace prior fitted per client
(`RegionalPrior`).

    from fdia_analysis.models.config import FederatedSettings
    from fdia_analysis.federated.localizer import FedBusCNN
    loc = FedBusCNN(federated=FederatedSettings(K=3)).fit(train, val=val)   # needs the [federated] extra
    rep = loc.score(test)
"""
