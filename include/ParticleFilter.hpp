/*
 * Based on:
 *     Probabilistic Robotics
 *     Sebastian Thruns
 */

#ifndef PARTICLE_FILTER_HPP
#define PARTICLE_FILTER_HPP

#include <ctime>
#include <boost/random.hpp>
#include <cmath>

namespace olamani {

namespace base {

template <unsigned int N>
struct Particle {
    double dimension[N];
    double weight;
}

template <unsigned int N, unsigned int P>
class ParticleFilter {
public:
    ParticleFilter();
    ~ParticleFilter();
    void initWithBelief(double mu[N], double dev[N]);
    void update(void(* weight)(Particle<N> &p));
    void predict(void(* predict)(Particle<N> &p));
    void resample();
    void getMean(int n);
    void getVariance(int n);
    double getFit();
    void computeParameters();
private:
    Particle<N> particles[P];
    double means[N];
    double variances[N];
    double w_slow;
    double w_fast;
    double a_slow;
    double a_fast;
    double w_avg;
    double mu[N];
    double dev[N];
    double total_w;
    double fit;
}

template <unsigned int N, unsigned int P>
ParticleFilter<N, P>::ParticleFilter() {
    for (int i = 0; i < N; ++i) {
        means[i] = 0.0;
        variances[i] = 0.0;
    }
    total_w = 1.0;
    fit = 0.0;
    w_slow = 0.0;
    w_fast = 0.0;
    a_slow = 0.1;
    a_fast = 2.0;
    w_avg = 0.0;
}

template <unsigned int N, unsigned int P>
ParticleFilter<N, P>::~ParticleFilter() {

}

template <unsigned int N, unsigned int P>
ParticleFilter<N, P>::initWithBelief(double mu[N], double dev[N]) {
    for (int i = 0; i < N; ++i) {
        this->mu[i] = mu[i];
        this->dev[i] = dev[i];
    }
    boost::mt19937 rng(time(0));
    for (int i = 0; i < N; ++i) {
        boost::uniform_real<> p_dist(0.0, 2.0 * dev[i]);
        for (int j = 0; j < P; ++j) {
            particles[j].dimension[i] = mu[i] - dev[i] + p_dist(rng);
        }
    }
    for (int i = 0; i < P; ++i) {
        particles[i].weight = 1.0 / P;
    }
}

template <unsigned int N, unsigned int P>
void ParticleFilter<N, P>::resample() {
	w_slow = w_slow + a_slow * (w_avg - w_slow);
	w_fast = w_fast + a_fast * (w_avg - w_fast);
	Particle<N> new_particles[P];
	boost::mt19937 rng(time(0));
	boost::uniform_real<> rdist(0.0, 1.0);
	double r = (1.0 / P) * rdist(rng);
	double c = particles[0].weight;
	int i = 1;
	for (int j = 0; j < P; ++j) {
		if (w_slow != 0.0 && (rdist(rng) < 1.0 - w_fast / w_slow)) {
			boost::uniform_real<> dist(0, 2.0 * dev[i]);
			Particle<N> p;
			for (int k = 0; k < N; ++k) {
				p.dimension[k] = means[k] - dev[k] + dist(rng);
			}
			new_particles[j] = p;
		} else {
			double u = r + ((double)j) * (1.0 / P);
			while (u > c) {
				i = (i + 1) % P;
				c += particles[i].weight;
			}
			new_particles[j] = particles[i];
		}
	}
	total_w = 0.0;
	for (i = 0; i < P; ++i) {
		particles[i] = new_particles[i];
		total_w += particles[i].weight;
	}
	fit = total_w;
	// Re-normalize
	double total_w_bu = 0.0;
	for (int i = 0; i < P; ++i) {
		particles[i].weight /= total_w;
		total_w_bu += particles[i].weight;
	}
	total_w = total_w_bu;
}

template <unsigned int N, unsigned int P>
void ParticleFilter<N, P>::predict(void(* predict)(Particle<N> &p)) {
	for (int i = 0; i < P; ++i) {
		predict(particles[i]);
	}
}

template <unsigned int N, unsigned int P>
void ParticleFilter<N, P>::update(void(* update)(Particle<N> &p)) {
	total_w = 0.0;
	for (int i = 0; i < P; ++i) {
		update(particles[i]);
		total_w += particles[i].weight;
	}
	fit = total_w;
	// Re-normalize
	double total_w_bu = 0.0;
	for (int i = 0; i < P; ++i) {
		particles[i].weight /= total_w;
		total_w_bu += particles[i].weight;
		w_avg += particles[i].weight / P;
	}
	total_w = total_w_bu;
}

template <unsigned int N, unsigned int P>
double ParticleFilter<N, P>::getMean(int n) {
	return means[n];
}

template <unsigned int N, unsigned int P>
double ParticleFilter<N, P>::getVariance(int n) {
	return variances[n];
}

template <unsigned int N, unsigned int P>
double ParticleFilter<N, P>::getFit() {
	return fit;
}

template <unsigned int N, unsigned int P>
void ParticleFilter<N, P>::computeParameters() {
	for (int i = 0; i < N; ++i) {
		means[i] = 0.0;
		variances[i] = 0.0;
		for (int j = 0; j < P; ++j) {
			means[i] += particles[j].dimension[i] * particles[j].weight;
		}
		means[i] /= total_w;
		variances[i] = 0.0;
		for (int j = 0; j < P; ++j) {
			variances[i] += pow(means[i] - particles[j].dimension[i], 2.0);
		}
		variances[i] /= (P - 1);
	}
}

} // namespace base

} // namespace olamani

#endif // PARTICLE_FILTER_HPP
