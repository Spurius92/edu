# Useful information about databases

## current position is [here](https://youtu.be/qw--VYLpxG4?list=PL_nqSomxocMzoZPxiZGs5fxudJxaXZwd9&t=6815)

Installed current version of it on Windows 10

Instead of `psql` command i have to use `runpsql`, because this is how the script called, and i don't want to mess with this system

## [Documentation](https://www.postgresql.org/docs/11/index.html))

## Takeaways

1. Created a user, password and already have some default database

2. The database management system is listening the 5432 port

3. To deal with the encoding problem on Windows had to do `\! chcp 1251` and now the console displays russian symbols just fine

4. `CREATE DATABASE person;` to create, `DROP DATABASE person;` to delete

5. `CREATE TABLE person;`, `DROP TABLE person;`

6. The columns in the database have to be specyfied type, and type puts constraint on what data can be stored and how much

    I guess, this is the reason why we need to make migrations in Django, after changing models

    Migrations let database apply changes and update its structure

7. [mockaroo.com](https://mockaroo.com/) for creating random data

8. `\i /destination/to/sql/file` makes a new table from the selected file

9. ### select unique values

    `SELECT DISTINCT last_name FROM person;`

10. ### select in order

    `SELECT last_name FROM person ORDER BY id DESC;`

11. ### select mupliple choices with AND

    `SELECT * FROM person WHERE gender='Female' AND (country_of_birth = 'Poland' OR country_of_birth='China');`

12. ### Comparison operator

    `SELECT 1 <> 1;` means '1 not equal to 1', which returns false
    others work too like `SELECT 1 < 2;`

13. ### returns first 10 rows of the table

    `SELECT * FROM person LIMIT 10;`

14. ### return everything starting from id=5

    `SELECT * FROM person OFFSET 5;`

15. ### select number of rows

    `OFFSET` means starting point to selection
    `SELECT * FROM person OFFSET 5 FETCH FIRST 5 ROW ONLY;`
    Alternatively can be done by `SELECT * FROM person OFFSET 5 LIMIT 5;`

16. ### select values from a list

    `SELECT * FROM person WHERE country_of_birth IN ('China', 'Brasil');`

    `SELECT * FROM person WHERE country_of_birth IN ('China', 'Brasil') ORDER BY first_name LIMIT 5;` works too

17. ### Select between dates

    `SELECT * FROM person WHERE date_of_birth BETWEEN DATE '1999-01-01' AND '1999-12-31';`

18. ### Wildcards

    `SELECT * FROM person WHERE email LIKE '%.com' LIMIT 5;`

    `SELECT * FROM person WHERE email LIKE '%google%' LIMIT 5;`

19. ### select emails that contain some number of characters

    7 underscores mean 7 chars before @ sign in the email
    `SELECT * FROM person WHERE email LIKE '_______@%';`

    select people whose first name contains exactly 7 characters
    `SELECT * FROM person WHERE first_name LIKE '_______';`

20. ### select country names starting with some char, ignoring the case

    Difference is `ILIKE` instead of `LIKE`
    `SELECT * FROM person WHERE first_name ILIKE 'p%';`

21. ### grouping by some feature

    `SELECT country_of_birth, COUNT(*) FROM person GROUP BY country_of_birth;`
    outputs something like this:
    `Poland                             |    28`
    `Democratic Republic of the Congo   |     1`
    `Costa Rica                         |     4`
    `Thailand                           |     3`

22. ### HAVING

    select countries that have more then 5 people in the previous result of grouping
    `SELECT country_of_birth, COUNT(*) FROM person GROUP BY country_of_birth HAVING COUNT(*) > 5 ORDER BY country_of_birth;`

23. ### MIN, MAX, AVG

    select minimal, maximal or average value from a column. Only for numeric
    `SELECT make, model, MAX(price) FROM car GROUP BY make, model;`

24. ### SUM

    `SELECT SUM(price) FROM car;`

25. ### Arithmetics

    Can perform arithmetic operations on columns. To do that just type the operation right away
    `SELECT 10!;`
    `SELECT price * .1 FROM car;`
    `SELECT ROUND(price * .1, 2) FROM car;`
    `SELECT ROUND(price * .1, 2) FROM car;`

26. ### Coalesce

    if some cells don't have values, impute with default value
    `SELECT COALESCE(email, 'Unknown') FROM person;`

27. ### Safe zero division

    If we try to divide by zero, postgres will throw a Zero division error.
    `SELECT 10 / 0;`
    But we can overcome this by doing `NULLIF(10, 10)`
    It will return 10 in this case, but if there is zero value, result will be empty.
    It is useful, if we need to perform potentially dangerous operations in our queries,
    that can end up with division by zero
    `SELECT COALESCE(10 / NULLIF(0, 0), 0);` will return 0
