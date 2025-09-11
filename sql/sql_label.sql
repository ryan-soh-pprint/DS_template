SELECT 
    t.lot_id,
    lot.name AS lot_name,
    property.name AS property_name,
    t.value AS property_value,
    lot.company_id,
    lot.product_type_id,
    product_type.name AS product_name
FROM (
    SELECT lot_id, numerical_property_id AS prop_id, value
    FROM lot_numerical_property_value
    UNION ALL
    SELECT lot_id, option_property_id AS prop_id, value
    FROM lot_option_property_value
) t
INNER JOIN lot ON lot.id = t.lot_id
INNER JOIN (
    SELECT id, property_id FROM numerical_property
    UNION ALL
    SELECT id, property_id FROM option_property
) p ON p.id = t.prop_id
INNER JOIN property ON property.id = p.property_id
INNER JOIN product_type ON product_type.id = lot.product_type_id
WHERE lot.company_id = 1374
  AND lot.product_type_id = 6907
  AND lot.name LIKE "%VR%"
ORDER BY lot_id;
