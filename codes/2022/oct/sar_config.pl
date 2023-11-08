#!/usr/bin/env perl

my $CONFIG_STR =
  '# Default settings for /etc/init.d/sysstat, /etc/cron.d/sysstat
# and /etc/cron.daily/sysstat files
#

# Should sadc collect system activity informations? Valid values
# are "true" and "false". Please do not put other values, they
# will be overwritten by debconf!
ENABLED="true"
';

my $FILE = '/etc/default/sysstat';
open my $fh, '>', $FILE or die "Can't open $FILE: $!";
print $fh $CONFIG_STR;
close $fh;

print "enable sysstat [OK]\n";

# ---------------------------------------------------------------------
$CONFIG_STR = '
# The first element of the path is a directory where the debian-sa1
# script is located
PATH=/usr/lib/sysstat:/usr/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# Activity reports every 10 minutes everyday
* * * * * root command -v debian-sa1 > /dev/null && debian-sa1 20 3

# Additional run at 23:59 to rotate the statistics file
59 23 * * * root command -v debian-sa1 > /dev/null && debian-sa1 60 2
';

$FILE = '/etc/cron.d/sysstat';
open $fh, '>', $FILE or die "Can't open $FILE: $!";
print $fh $CONFIG_STR;
close $fh;

print "setting report frequency [OK] \n";

# ---------------------------------------------------------------------
$FILE       = '/etc/sysstat/sysstat';
$CONFIG_STR = '# sysstat configuration file. See sysstat(5) manual page.

# How long to keep log files (in days).
# Used by sa2(8) script
# If value is greater than 28, then use sadc\'s option -D to prevent older
# data files from being overwritten. See sadc(8) and sysstat(5) manual pages.
HISTORY=28

# Compress (using xz, gzip or bzip2) sa and sar files older than (in days):
COMPRESSAFTER=10

# Parameters for the system activity data collector (see sadc(8) manual page)
# which are used for the generation of log files.
# By default contains the `-S DISK\' option responsible for generating disk
# statisitcs. Use `-S XALL\' to collect all available statistics.
SADC_OPTIONS="-S DISK"

# Directory where sa and sar files are saved. The directory must exist.
SA_DIR=/var/log/sysstat

# Compression program to use.
ZIP="xz"

# By default sa2 script generates yesterday\'s summary, since the cron job
# usually runs right after midnight. If you want sa2 to generate the summary
# of the same day (for example when cron job runs at 23:53) set this variable.
#YESTERDAY=no

# By default sa2 script generates reports files (the so called sarDD files).
# Set this variable to false to disable reports generation.
#REPORTS=false
';

open $fh, '>', $FILE or die "Can't open $FILE: $!";
print $fh $CONFIG_STR;
close $fh;
print "setting sysstat history [OK] \n";

# ---------------------------------------------------------------------

`systemctl restart sysstat`;
print "restart sysstat [OK] \n";
