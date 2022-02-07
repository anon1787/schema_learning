#!/usr/bin/perl

@cols = qw(p_partkey p_name p_mfgr p_brand p_type p_size p_container p_comment p_retailprice s_suppkey s_name s_address s_phone s_acctbal s_comment c_custkey c_name c_address c_phone c_acctbal c_mktsegment c_comment n_nationkey n_name n_comment r_regionkey r_name r_comment o_orderkey o_orderstatus o_totalprice o_orderdate o_orderpriority o_shippriority o_clerk o_comment ps_availqty ps_supplycost ps_comment l_linenumber l_quantity l_extendedprice l_discount l_tax l_returnflag l_linestatus l_shipdate l_commitdate l_receiptdate l_shipinstruct l_shipmode l_comment);


foreach $i (0..$#cols) {
    $hash{"column$i"} = $cols[$i];
}

foreach $k (keys(%hash)) {
    print $hash{$k};
}

$re = (join "|", map {"(column$_)"} (1..scalar(@cols)));
#$re = join "|", @cols;
print $re,"\n";
while(<>) {
     s/\b($re)\b/$hash{$1}/eeg;
     print;
}
