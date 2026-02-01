#pragma once

struct Cooldown
{
	float target;
	float value;

	Cooldown()
		: target(0.0f)
		, value(0.0f)
	{}

    explicit
	Cooldown(float delay)
		: target(delay)
		, value(0.0f)
	{}

	Cooldown(float delay, float current_time)
		: target(delay)
		, value(current_time)
	{}

	void update(float dt)
	{
		value += dt;
	}

	bool updateAutoReset(float dt)
	{
		update(dt);
		bool res = ready();
		if (res) {
			reset();
		}
		return res;
	}

    [[nodiscard]]
	bool ready() const
	{
		return value >= target;
	}

    [[nodiscard]]
	bool readyNext(float dt) const
	{
		return value < target && value + dt >= target;
	}

    [[nodiscard]]
    float getRatio() const
    {
        return value / target;
    }

	void reset()
	{
		value = 0.0f;
	}
};
