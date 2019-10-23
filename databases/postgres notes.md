# Useful information about databases

[tutorial](https://www.youtube.com/watch?v=qw--VYLpxG4&list=PL_nqSomxocMzoZPxiZGs5fxudJxaXZwd9&index=19&t=5131s)

on YouTube channel Freecodecamp about Postgres

Installed current version of it on Windows 10

Instead of `psql` command i have to use `runpsql`, because this is how the script called, and i don't want to mess with this system

At this point, nothing more that i didn't know yet:

1. Created a user, password and already have some default database

2. The database management system is listening the 5432 port

3. To deal with the encoding problem on Windows had to do `\! chcp 1251` and now the console displays russian symbols just fine

4. `CREATE DATABASE person;` to create, `DROP DATABASE person;` to delete

5. `CREATE TABLE person;`, `DROP TABLE person;`

6. The columns in the database have to be specyfied type, and type puts constraint on what data can be stored and how much

    I guess, this is the reason why we need to make migrations in Django, after changing models

    Migrations let database apply changes and update its structure

7. Documentation for postgres can be found [here](https://www.postgresql.org/docs/11/index.html)

8. [mockaroo.com](https://mockaroo.com/) for creating random data

9. `\i /destination/to/sql/file` makes a new table from the selected file

10. `SELECT DISTINCT last_name FROM person;` - to return unique values

11. `SELECT last_name FROM person ORDER BY id DESC;`

12. `SELECT * FROM person WHERE gender='Female' AND (country_of_birth = 'Poland' OR country_of_birth='China');`

13. Comparison operator `SELECT 1 <> 1;` means '1 not equal to 1', which returns false

14. `SELECT * FROM person LIMIT 10;` returns first 10 rows of the table

15. `SELECT * FROM person OFFSET 5;` return everything starting from id=5

16. `SELECT * FROM person OFFSET 5 FETCH FIRST 5 ROW ONLY;` select ids from 6 to 10
    Alternatively can be done by `SELECT * FROM person OFFSET 5 LIMIT 5;`

## Current position: <https://youtu.be/qw--VYLpxG4?list=PL_nqSomxocMzoZPxiZGs5fxudJxaXZwd9&t=5564>
