package minicpbp.examples;

import static org.junit.Assert.assertEquals;

import java.lang.reflect.Field;
import java.lang.reflect.Method;

import org.junit.Test;

public class NavigationServiceTest {

    @Test
    public void rewardMatrixMatchesRddlCurrentStateReward() throws Exception {
        Field goal = NavigationService.class.getDeclaredField("goalPosIdx");
        goal.setAccessible(true);
        goal.setInt(null, 2);

        Method builder = NavigationService.class.getDeclaredMethod(
            "buildRewardMatrix", int.class);
        builder.setAccessible(true);
        int[][][] rewards = (int[][][]) builder.invoke(null, 4);

        // Entering the goal still receives the current non-goal state's -1.
        assertEquals(-1, rewards[0][0][2]);
        // The goal is absorbing with zero reward thereafter.
        assertEquals(0, rewards[2][0][2]);
        // The dead state remains penalized.
        assertEquals(-1, rewards[3][0][3]);
    }
}
