# Notes about postgres setup process on Ubuntu server 18.04

## Installation

* `sudo apt-get install postgresql postgresql-contrib` This installs a server on the machine, so we can store the database on it

## Login, users, roles et cetera

name postgres is default for the role, user and database here, so it is important. We can add user with following

* `sudo -i -u postgres` launches interactive postgres shell

  ** from which we can run `psql` command with the active user `postgres`

  ** Alternatively, just `sudo -u postgres psql` to do the same

* In the postgres shell run `createuser --interactive` and type user name

* Create database with `createdb <name of db>`

* `psql -d <dbname>` from the main shell to connect with <dbname> but user stays the same

* `sudo -u <username> psql` connect to database with <username>

Done! Now we can access `psql` and manage the databases

## Useful commands inside the psql

* `\conninfo` to view the information about activedb, user, socket and port

* `\du` list of users

* `\dt` list of tables
