#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include "device.h"

class DeviceManager
{
public:
    static DeviceManager &get_instance()
    {
        static DeviceManager instance;
        return instance;
    }

    DeviceManager(DeviceManager const &) = delete;
    void operator=(DeviceManager const &) = delete;

    DeviceType get_current_device() const
    {
        return current_device;
    }

    void set_current_device(DeviceType device)
    {
        this->current_device = device;
    }

private:
    DeviceType current_device;
    // Singleton pattern
    DeviceManager() : current_device(DeviceType::CPU) {}
    DeviceManager(DeviceManager const &) = delete;
    void operator=(DeviceManager const &) = delete;
};

#endif // DEVICE_MANAGER_H
