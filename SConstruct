import os

home = os.environ['HOME']
build_folder = 'Documents/scipy2010'
build_directory = os.path.join(home, build_folder)

env = Environment(ENV=os.environ)
env['PDFLATEXFLAGS'] = ['-interaction=nonstopmode', '-recorder', '--shell-escape']

Export('env')
SConscript(['SConscript'],build_dir=build_directory)
