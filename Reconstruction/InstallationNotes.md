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


If the pictures are not already in jpg format:
for f in *.png
do
convert $f $f.jpg
rm $f
done

To run bundler, navigate to a folder with pictures in jpg format:
./RunBundler.sh


Then:
./bin/Bundler2PMVS list.txt bundle/bundle.out

Edit pmvs/prep_pmvs.sh:
BUNDLER_BIN_PATH="/home/bce/CV/bundler/bin"

Then run it
bash pmvs/pmvs_prep.sh




PMVS:

Download the fixed version of pmvs (pmvs-2fix0)
Download clapack.h and f2c.h from http://www.netlib.org/clapack/
Copy these files to pmvs/numerics
Edit the include lines in mylapack.cc to read:
extern "C" {
    #include "f2c.h"
    #include "clapack.h"
}

Edit patchOrganizerS.cc around line 130 (add ofstr.close()):

      ofstr << patch << endl;
    }
    ofstr.close();
  }

  {
    char buffer[1024];


If you compile now, it will not produce any output. You must modify the last lines in findMatch.cc to be (writePatches2 -> writePatches):
void CfindMatch::write(const std::string prefix) {
  m_pos.writePatches();
}

Finally, make it:

make clean
make depend
make

Run it!

pmvs2 pmvs/ pmvs/pmvs_options.txt


<BAD SOLUTION>
Installing PMVS - install the unpatched version, which contains binaries
The libjpeg binary is problematic, but remaking the whole package causes a problem with lapack. Instead:


Share folder to VM:
shut down VM, add folder path on VirtualBox Settings
Restart VM, navigate to /media/[folder]
To get ownership of file, run
sudo chown bce: [folder]
