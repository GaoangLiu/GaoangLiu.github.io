
-- After the first pass, @rownum will contain the total number of rows. 
-- This can be used to determine the median, so no second pass or join is needed.
SELECT AVG(dd.unitprice) as median
FROM (
    SELECT d.unitprice, @rownum := @rownum+1 as `row_number`, @total_rows:=@rownum
    FROM orderdetails d, (SELECT @rownum:=0) r
        WHERE d.unitprice is NOT NULL
        -- put some where clause here
        ORDER BY d.unitprice
) AS dd
    WHERE dd.row_number IN ( FLOOR((@total_rows+1)/2), FLOOR((@total_rows+2)/2) );