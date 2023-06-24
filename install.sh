
# Create an conda environment.
conda create -n date python=3.10 -y \
&& conda activate date \
&& conda install -c "nvidia/label/cuda-11.7.0" cuda-nvcc -y \
&& conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia -y \
&& pip install -U openmim \
&& mim install 'mmcv-full==1.7.0' \
&& pip install 'mmdet==2.25.1';

if [ $? ]
then
    python3 -m pip install --user -e .
else
    echo "Installation DATE failed."
fi

echo "Installation finished, please link your data (e.g., MSCOCO) under ./data"
