install gfortran
install liblapack
export LD_LIBRARY_PATH=/home/bce/CV/bundler/bin >> ~/.bashrc



Lowe keypoint descriptor doesn't work (built for 32 bit machines)
sudo -i
cd /etc/apt/sources.list.d
echo "deb http://old-releases.ubuntu.com/ubuntu/ raring main restricted universe multiverse" >ia32-libs-raring.list
apt-get update
apt-get install ia32-libs

copy sift from lowe package to ~/bundler/bin

apt-get install meshlab


Installing PMVS - install the unpatched version, which contains binaries
The libjpeg binary is problematic, but remaking the whole package causes a problem with lapack. Instead:

tar xf pmvs-2.tar.gz
cd pmvs-2/program/main/
cp mylapack.o mylapack.o.backup
make clean
cp mylapack.o.backup mylapack.o
make depend
make


To run bundler, navigate to a folder with pictures and run ./RunBundler.sh
Then:
./bin/Bundler2PMVS list.txt bundle/bundle.out

Edit pmvs/prep_pmvs:
BUNDLER_BIN_PATH="/home/bce/CV/bundler/bin"



PMVS:

pmvs2 pmvs/ pmvs/pmvs_options.txt
