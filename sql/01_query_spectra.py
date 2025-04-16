import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from pptoolbox.connectors import BaseEFSConnector, PFDBConnector

from datetime import datetime

load_dotenv(find_dotenv())

PF_SQL_PASSWORD = os.environ.get("PLATFORM_SQL_PASSWORD", None)  # PP_SQL_PASSWORD
PF_KEY_PATH = os.environ.get("DS_SERVER_KEYPATH", None)  # PP_SERVER_KEYPATH
PF_EFS_URL = os.environ.get("PF_EFS_URL", None)  # PP_EFS_URL

print(PF_SQL_PASSWORD,PF_KEY_PATH,PF_EFS_URL)

# samples up to 240927
BASE_QUERY = """
SELECT
	sp.id AS specimen_id,
	l.id AS lot_id,
	l.name AS lot_name,
	sp.date_scanned,
	sp.analyzer_id AS analyser_id,
    l.company_id as company_id,
	p.id AS product_id,
    p.name AS product_name
FROM
	specimen sp
	INNER JOIN lot l ON l.id = sp.lot_id
	INNER JOIN product_type p on l.product_type_id = p.id
WHERE l.company_id = 1303
ORDER BY l.name;
"""

if __name__ == "__main__":
    SPECTRA_COLS = [
        "raw_data",
        "dark_ref_data",
        "white_ref_data",
        "dark_ref_scan_time",
        "white_ref_scan_time",
    ]
    db_conn = PFDBConnector()
    info_df = db_conn.query(PF_KEY_PATH, PF_SQL_PASSWORD, BASE_QUERY)
    print("successful query")
    efs_conn = BaseEFSConnector(url=PF_EFS_URL)
    print(len(info_df))
    spectra_df = efs_conn.fetch_spectra(info_df.specimen_id.values)
    joined_df = info_df.merge(
        spectra_df.loc[:, SPECTRA_COLS], left_on="specimen_id", right_index=True
    )
    joined_df.set_index("lot_id", inplace=True)
    print("Successfully queried from DB")
    print(joined_df.head(n=10))
    print(joined_df.shape)

    today_date = datetime.now().strftime("%y%m%d")

    datafolder_path = Path ("data")
    raw_folder = datafolder_path / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_folder / f"spectra_{today_date}.csv"
    joined_df.to_csv(raw_csv)
