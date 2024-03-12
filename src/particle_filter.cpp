/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

std::default_random_engine gen; // Random engine

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 10;  // TODO: Set the number of particles
  
  // Create normal distributions for x, y, and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);

  // Initialize particles
  for (int i = 0; i < num_particles; ++i) {
    Particle particle;
    
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = fmod(dist_theta(gen), 2 * M_PI);
    particle.weight = 1.0;

    particles.push_back(particle);
  }
  
  is_initialized = true; // Set is_initialized to true after initialization
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::normal_distribution<double> dist_x(0, std_pos[0]);
  std::normal_distribution<double> dist_y(0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0, std_pos[2]);

  for (unsigned int i=0; i<particles.size(); ++i) {
    double A = particles[i].theta + yaw_rate * delta_t;
    
    if (fabs(yaw_rate) < 0.00001) { // Check for zero yaw rate to avoid division by zero in calculations
      particles[i].x += velocity * delta_t * cos(particles[i].theta);
      particles[i].y += velocity * delta_t * sin(particles[i].theta);
      // particle.theta remains unchanged if yaw_rate is zero
    } else {
      particles[i].x += (velocity / yaw_rate) * (sin(A) - sin(particles[i].theta));
      particles[i].y += (velocity / yaw_rate) * (cos(particles[i].theta) - cos(A));
      particles[i].theta = A;
    }

    // Add noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  double dx, dy, distance_square;
  
  for (unsigned int i=0; i<observations.size(); ++i) {
    double min_distance_square = std::numeric_limits<double>::max();
    int nearest_landmark_id = -1;
    
    for (unsigned int j=0; j<predicted.size(); ++j) {
      dx = observations[i].x - predicted[j].x;
      dy = observations[i].y - predicted[j].y;
      distance_square = dx * dx + dy * dy;
      
      // find the minimum distance square instead of distance for performance consiferation
      // eliminate the function sqrt() to find the nearest landmark
      if (distance_square < min_distance_square) {
        min_distance_square = distance_square;
        nearest_landmark_id = predicted[j].id;
      }
    }
    
    // update the associated landmark id to the observation
    observations[i].id = nearest_landmark_id;
  }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  for (unsigned int i=0; i<particles.size(); ++i) {
    // Filter landmarks within the sensor_range
    std::vector<LandmarkObs> in_range_landmarks;
    for (unsigned int j=0; j<map_landmarks.landmark_list.size(); ++j) {
      // double distance = sqrt(pow(particle.x - landmark.x_f, 2) + pow(particle.y - landmark.y_f, 2));
      // treat sensor boudary as rectangle instead of circle for performance reason
      if ((fabs(particles[i].x - map_landmarks.landmark_list[j].x_f) <= sensor_range) && 
          (fabs(particles[i].y - map_landmarks.landmark_list[j].y_f) <= sensor_range)) {
        LandmarkObs temp;
        temp.id = map_landmarks.landmark_list[j].id_i;
        temp.x  = map_landmarks.landmark_list[j].x_f;
        temp.y  = map_landmarks.landmark_list[j].y_f;
        in_range_landmarks.push_back(temp);
      }
    }
    
    // Transformation of the all observations from car coordinate system to map coordinate
    std::vector<LandmarkObs> transformed_obs;
    for (unsigned int j=0; j<observations.size(); ++j) {
      LandmarkObs temp;
      temp.x = 	cos(particles[i].theta) * observations[j].x - 
        		sin(particles[i].theta) * observations[j].y + particles[i].x;
      temp.y = 	sin(particles[i].theta) * observations[j].x + 
        		cos(particles[i].theta) * observations[j].y + particles[i].y;
      transformed_obs.push_back(temp);
    }
    
    // Associating the nearest landmarks to the observations
    dataAssociation(in_range_landmarks, transformed_obs);
    
    // Calculate the multivariate guassian distribution of each particle based on observations
    particles[i].weight = 1.0;
    for (unsigned int j=0; j<transformed_obs.size(); ++j){
      // Assuming No observation does not associate to a landmark
      unsigned int k;
      for (k=0; k<in_range_landmarks.size(); ++k) {
        if (in_range_landmarks[k].id == transformed_obs[j].id) {
          break; //Found the matching landmark
        }
      }
      if (k < in_range_landmarks.size()) {
        // P(x,y) = 1/(2*pi*sigma_x*sigma_y) * exp(-(((x-mu_x)^2/(2*sigma_x^2)) + ((y-mu_y)^2/(2*sigma_y^2))))
        // Using the above formula to calculate the probability of each observation
        double dx = fabs(transformed_obs[j].x - in_range_landmarks[k].x);
        double dy = fabs(transformed_obs[j].y - in_range_landmarks[k].y);
        double x_term = dx * dx / (2 * std_landmark[0] * std_landmark[0]);
        double y_term = dy * dy / (2 * std_landmark[1] * std_landmark[1]);
        double p_xy   = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
        // The product of all observation will be the weight of the particle
        if (p_xy == 0) {
          particles[i].weight *= 0.001;
        }
        else {
          particles[i].weight *= p_xy;
        }
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<double> w; //vector of weights of all particles
  for (const auto& p : particles) {
    w.push_back(p.weight);
  }
  
  // Create the decrete distribution which use particle's weight as the probability for resampling
  std::discrete_distribution<int> weighted_dist(w.begin(), w.end());
  
  // Create resampled particles
  std::vector<Particle> resampled_particles;
  for (int i=0; i < num_particles; ++i) {
    int index = weighted_dist(gen); // Generate index based on particles' weight
    resampled_particles.push_back(particles[index]);
  }
  
  particles = resampled_particles; // replace old particles with resampled one
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}