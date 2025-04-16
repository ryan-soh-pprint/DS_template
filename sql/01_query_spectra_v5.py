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
WHERE l.company_id = 1313
ORDER BY l.name;
"""

if __name__ == "__main__":
 
    db_conn = PFDBConnector()
    info_df = db_conn.query(PF_KEY_PATH, PF_SQL_PASSWORD, BASE_QUERY)
    print("Successfully queried from DB")
    if input(f'Found {info_df.shape[0]} rows. Proceed? [y]/n ') == 'n':
        print('stopping data pull')
        exit() 
    else:
        pass
    efs_conn = BaseEFSConnector(url=PF_EFS_URL)
    spectra_df = efs_conn.fetch_spectra(info_df.specimen_id.values)
    column_diff = set(info_df.columns) & set(spectra_df.columns)
    joined_df = spectra_df.merge(
        info_df.drop(column_diff, axis=1), right_on="specimen_id", left_index=True
    )
    joined_df.set_index("lot_id", inplace=True)
    # Rename the qmini, qneo columns to align with automl_v5
    mapper = {
        "calc_data_mini": "raw_data_mini",
        "calc_data_neo": "raw_data_neo",
    }
    joined_df.rename(mapper=mapper, axis=1, inplace=True)
    print(joined_df.head(n=10)) 

    today_date = datetime.now().strftime("%y%m%d")

    datafolder_path = Path ("data")
    raw_folder = datafolder_path / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_folder / f"spectra_{today_date}.csv"
    joined_df.to_csv(raw_csv)
