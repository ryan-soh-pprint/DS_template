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

batches = [4720, 4799]
# samples in batch 4720
BASE_QUERY = f"""
WITH BatchLots AS (
    SELECT
        l.id AS lot_id,
        l.name AS lot_name
    FROM
        lot l
        INNER JOIN lot_batch_bridge br ON br.lot_id = l.id
    WHERE
        br.lot_batch_id IN (4720, 4799)
)
SELECT 
    lopv.lot_id, 
    lot.name AS lot_name, 
    property.name AS property_name,
    lopv.value AS property_value,
    lot.company_id,
    lot.product_type_id,
    product_type.name AS product_name
FROM 
    lot_option_property_value lopv
INNER JOIN 
    lot ON lot.id = lopv.lot_id
INNER JOIN 
    option_property ON option_property.id = lopv.option_property_id
INNER JOIN 
    property ON option_property.property_id = property.id
INNER JOIN 
    product_type ON product_type.id = lot.product_type_id
WHERE 
    lopv.lot_id IN (SELECT lot_id FROM BatchLots)
ORDER BY 
    lot.name;
"""

if __name__ == "__main__":

    db_conn = PFDBConnector()
    info_df = db_conn.query(PF_KEY_PATH, PF_SQL_PASSWORD, BASE_QUERY).set_index("lot_id")
    print("successful query")
    today_date = datetime.now().strftime("%y%m%d")

    datafolder_path = Path ("data")
    raw_folder = datafolder_path / "raw"
    raw_folder.mkdir(parents=True, exist_ok=True)
    raw_csv = raw_folder / f"info_cat_batch_{today_date}.csv"
    info_df.to_csv(raw_csv)
