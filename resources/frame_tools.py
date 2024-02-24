import cv2
import math
import random


from pathlib import Path


def calculate_sample_size(
        population_size:int,
        margin_of_error:float=0.05,
        estimated_proportion:float=0.5
) -> int:
    '''Returns a sample size for a given population_size with 95% confidence

    How many random frames should be taken from a video in order to get a repre-
    sentative sample? This method is a simple implementation for calculating the
    sample size in order to get a margin of error of 5%, considering a normal 
    distribution in a population with known size. If the sample_size is negative
    gets a sample of 30% the frame population.

    Args:
        population_size (int): the number of frames in the video
        margin_of_error (float): decimal based, margin of error
        estimated_proportion (float): point where the error is higher or a known
        value for the population distribution
    
    Returns:
        int: the number of random frames to be taken from the video
    '''

    # for a 95% confidence level
    z_score = 1.96  
    sample_size = (
        z_score**2 * estimated_proportion * (1 - estimated_proportion)
    ) / (margin_of_error**2)

    # population correction factor    
    sample_size_adjusted = sample_size * (
        (population_size - sample_size) / (population_size - 1)
    ) if population_size < float('inf') else sample_size
    
    if sample_size_adjusted < 0:
        sample_size_adjusted = 0.3 * population_size
        
    return math.ceil(sample_size_adjusted)


def get_random_frames(
        input_path:Path,
        output_path:Path,
        source_name:str
) -> None:
    '''Return a list of n random frames using the calculate_sample_size method

    
    Args:
        input_path (Path): path string object to the video input
        output_paht (Path): path string object to the output folder in which to
        write the images
        source_name (str): pattern name used to save the images

    Returns:
        None  
    '''

    cap = cv2.VideoCapture(input_path)

    # Find the population of frames
    population = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculates sample size
    sample_size = calculate_sample_size(population_size=population)
    
    print(f'population:{population}\nsample size: {sample_size}')
    
    # set of unique frame indexes
    random_frame_indices = set(random.sample(range(population), sample_size))
    
    for frame_index in random_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if ret:

            # writes each image in the output folder
            cv2.imwrite(
                f'{output_path}/img_{frame_index}_{source_name}.jpg', frame
            )
    cap.release()
    return


def main():
    print(calculate_sample_size(population_size=10_000))


if __name__=='__main__':
    main()