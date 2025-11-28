python example_subject_full.py --setting clock --num_images 50 > evaluation_massive/omini_subject/clock_full.log
python example_subject_full.py --setting penguin --num_images 50 > evaluation_massive/omini_subject/penguin_full.log
python example_subject_full.py --setting oranges --num_images 50 > evaluation_massive/omini_subject/oranges_full.log

python example_subject_tea.py --setting clock --num_images 50 --rho 0.3 --alpha 0.5 > evaluation_massive/omini_subject_tea/clock_full.log
python example_subject_tea.py --setting penguin --num_images 50 --rho 0.3 --alpha 0.5 > evaluation_massive/omini_subject_tea/penguin_full.log
python example_subject_tea.py --setting oranges --num_images 50 --rho 0.3 --alpha 0.5 > evaluation_massive/omini_subject_tea/oranges_full.log
