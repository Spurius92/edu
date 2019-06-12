# Useful information about databases

June 12, started watching the tutorial on YouTube channel Freecodecamp about Postgres

Installed current version of it on Windows 10

Instead of `psql` command i have to use `runpsql`, because this is how the script called, and i don't want to mess with this system

At this point, nothing more that i didn't know yet:

1. Created a user, password and already have some default database

2. The database management system is listening the 5432 port

3. `CREATE DATABASE` to create, `DROP DATABASE` to delete

4. The columns in the database have to be specyfied type, and type puts constraint on what data can be stored and how much

    I guess, this is the reason why we need to make migrations in Django, after changing models

    Migrations let database apply changes and update its structure

5. Documentation for postgres can be found [here](https://www.postgresql.org/docs/11/index.html)
