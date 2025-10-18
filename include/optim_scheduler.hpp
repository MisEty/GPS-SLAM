#ifndef OPTIM_SCHEDULER
#define OPTIM_SCHEDULER

#include <iostream>
#include <torch/torch.h>

class OptimScheduler
{
public:
    OptimScheduler(torch::optim::Optimizer *optimizer, double gamma)
        : optimizer_(optimizer), gamma_(gamma) {}

    void step()
    {
        double lr = optimizer_->defaults().get_lr();
        optimizer_->defaults().set_lr(lr * gamma_);
    }

private:
    torch::optim::Optimizer *optimizer_;
    double gamma_;
};

#endif