import os

from dotenv import find_dotenv, load_dotenv

from pptoolbox.connectors import BaseEFSConnector, PFDBConnector

load_dotenv(find_dotenv())

PF_SQL_PASSWORD = os.environ.get("PLATFORM_SQL_PASSWORD", None)  # CN_SQL_PASSWORD
PF_KEY_PATH = os.environ.get("DS_SERVER_KEYPATH", None)  # CN_SERVER_KEYPATH
PF_EFS_URL = os.environ.get("PF_EFS_URL", None)  # CN_EFS_URL

# batch = 4720
BASE_QUERY = f"""
SELECT
	sp.id AS specimen_id,
	l.id AS lot_id,
	l.name AS lot_name,
	sp.date_scanned,
	sp.analyzer_id AS analyzer_id
FROM
	specimen sp
	INNER JOIN lot l ON l.id = sp.lot_id
    INNER JOIN lot_batch_bridge br on br.lot_id = l.id
WHERE
	# l.company_id = 1243
    br.lot_batch_id IN (4720, 4799)

ORDER BY
	l.created_dt;
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

    info_df = db_conn.query(PF_KEY_PATH, PF_SQL_PASSWORD, BASE_QUERY)    
    efs_conn = BaseEFSConnector(url=PF_EFS_URL)
    spectra_df = efs_conn.fetch_spectra(info_df.specimen_id.values)
    column_diff = set(info_df.columns) & set(spectra_df.columns)
    joined_df = spectra_df.merge(
        info_df.drop(column_diff, axis=1), right_on="specimen_id", left_index=True
    )
    joined_df.set_index("lot_id", inplace=True)


    from pathlib import Path
    from datetime import datetime
    today_date = datetime.now().strftime("%y%m%d")

    datafolder_path = Path ("data/raw")
    datafolder_path.mkdir(parents=True, exist_ok=True)
    raw_csv = datafolder_path / f"spectra_batch_{today_date}.csv"
    joined_df.to_csv(raw_csv)

