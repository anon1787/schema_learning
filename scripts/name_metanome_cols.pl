#!/usr/bin/perl

($colfile) = pop @ARGV;

open COLFILE, $colfile or die "$colfile";
$header = <COLFILE>;
chomp $header;
#print $header;

@cols = map {s/[\"\']//g; $_} (split /,/, $header);
#print join ":", @cols;
#print "\n";
#print $header;
#print scalar(@cols), "\n\n";

foreach $i (0..$#cols) {
    $hash{"column$i"} = $cols[$i];
}

foreach $k (keys(%hash)) {
#    print "dict: $hash{$k}";
}

$re = (join "|", map {"(column$_)"} (1..scalar(@cols)));
#$re = join "|", @cols;
#print $re,"\n";
while(<>) {
     s/\b($re)\b/$hash{$1}/eeg;
     print;
}
