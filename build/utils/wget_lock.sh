#!/bin/bash

# Define a function to generate a lock file name from the URL
generate_lockfile() {
    echo "/tmp/wget_$(echo "$1" | md5sum | cut -d ' ' -f 1).lock"
}

# Extract the URL (assuming it's the last argument for simplicity)
# Adjust as needed based on your typical wget arguments
URL="${@: -1}"

LOCKFILE=$(generate_lockfile "$URL")

lock_failed() {
    echo "Another wget instance is currently downloading $URL. Exiting."
    exit 1
}

# Try to acquire the lock and run wget with passed arguments
(
    # Wait for the lock for 5 seconds. If cannot acquire, run lock_failed function
    flock -w 5 200 || lock_failed

    # Run wget with all passed arguments
    wget "$@"

) 200>"$LOCKFILE"
