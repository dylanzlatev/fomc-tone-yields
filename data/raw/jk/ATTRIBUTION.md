# Attribution

The file `shocks_fed_jk_t.csv` is the updated monetary policy and central bank
information shock series of

Jarocinski, Marek, and Peter Karadi (2020). "Deconstructing Monetary Policy
Surprises - The Role of Information Shocks." American Economic Journal:
Macroeconomics 12(2): 1-43. doi:10.1257/mac.20180090

Source repository: https://github.com/marekjarocinski/jkshocks_update_fed
(file `shocks_fed_jk_t.csv`, fetched from the `main` branch; see
`data/raw/MANIFEST.csv` for the retrieval date and SHA-256).

License: the source data and final dataset are licensed under the Creative Commons
Attribution 4.0 International License (CC BY 4.0),
https://creativecommons.org/licenses/by/4.0/. The authors' code is BSD 3-Clause.
The file is unmodified here; unit conversion (percentage points to basis points)
is done downstream in `src/ftyields/data/surprises.py`.
