#!/bin/bash
USER_ID=${LOCAL_USER_ID:-9001}
echo 'Starting with username : markpp and UID : $USER_ID'
useradd -s /bin/bash -u $USER_ID -o -c '' -m markpp
export HOME=/home/markpp
su markpp bash -c 'bash'