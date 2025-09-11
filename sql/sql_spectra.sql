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
WHERE l.company_id = 1374
AND p.id = 6907 AND l.name LIKE "%VR%"
ORDER BY l.id;