# Data

Raw NHANES data files are not stored in this repository due to file size and
CDC/NCHS redistribution terms.

## Automatic Download

The analysis script (`Analysis.py`) will attempt to
download the required files automatically from the CDC/NCHS website when run.

## Manual Download

If automated download is blocked in your environment, download the following
eight `.XPT` files from the CDC NHANES website and place them in this folder:

| File | Cycle | Component | URL |
|---|---|---|---|
| `DEMO_J.XPT` | 2017–2018 | Demographics | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT |
| `PFAS_J.XPT` | 2017–2018 | Laboratory | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/PFAS_J.XPT |
| `BMX_J.XPT` | 2017–2018 | Body Measures | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT |
| `SMQ_J.XPT` | 2017–2018 | Smoking | https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT |
| `DEMO_L.XPT` | 2021–2023 | Demographics | https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/DEMO_L.XPT |
| `PFAS_L.XPT` | 2021–2023 | Laboratory | https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/PFAS_L.XPT |
| `BMX_L.XPT` | 2021–2023 | Body Measures | https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/BMX_L.XPT |
| `SMQ_L.XPT` | 2021–2023 | Smoking | https://wwwn.cdc.gov/Nchs/Nhanes/2021-2023/SMQ_L.XPT |

All files are in the public domain and freely available from:
https://www.cdc.gov/nchs/nhanes/