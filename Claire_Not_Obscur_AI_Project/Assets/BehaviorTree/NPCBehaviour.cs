using UnityEngine;
using System.Collections;

public class NPCBehaviour : MonoBehaviour
{
    public State state;

    [SerializeField] Movement enemy;
    [SerializeField] Movement myMovement;
    private float startTime = 0f;
    public enum State
    {
        Idle,
        Evade,
        Attack,
        Move,
        Hit
    }

    public enum MoveOptions
    {
        Dash,
        Walk,
        Jump
    }

    // Start is called once before the first execution of Update after the MonoBehaviour is created
    void Start()
    {

        state = State.Idle;
    }

    // Update is called once per frame
    void Update()
    {

        startTime += 1f/30f;
        if(startTime > 0.3f)
        {
            startTime = 0f;
            state = State.Idle;
            {
                myMovement.rightPressed = false;
                myMovement.leftPressed = false;
                myMovement.jumpPressed = false;
                myMovement.dashPressed = false;
                myMovement.attackPressed = false;
                myMovement.downPressed = false;
                if (enemy.attacking && Random.value > 0.7f && myMovement.IsGrounded())
                {
                    state = State.Evade;
                }

                else if (Mathf.Abs(enemy.transform.position.x - this.transform.position.x) < 1f && Random.value > 0.5f)
                {
                    state = State.Attack;
                }

                else if (Mathf.Abs(enemy.transform.position.x - this.transform.position.x) >= 0.2f)
                {
                    state = State.Move;
                }

                Debug.Log(state);
            }
            switch (state)
            {
                case State.Idle:
                    // Do nothing
                    break;
                case State.Evade:
                    Evade();
                    break;
                case State.Attack:
                    Attack();
                    break;
                case State.Move:
                    Move();
                    break;
                case State.Hit:
                    IsHit();
                    break;
            }
        }

    }

    public void Evade()
    {
        if (Random.value > 0.5f)
            JumpBack();
        else
            BackDash();
    }
    public void JumpBack()
    {
        if(transform.position.x > enemy.transform.position.x)
            myMovement.leftPressed = true;
        else
            myMovement.rightPressed = true;
        myMovement.jumpPressed = true;
        state = State.Idle;
    }

    public void BackDash()
    {
        if (transform.position.x > enemy.transform.position.x)
            myMovement.leftPressed = true;
        else
            myMovement.rightPressed = true;
        myMovement.dashPressed = true;
        state = State.Idle;
    }

    public void Attack()
    {
        myMovement.attackPressed = true;
        state = State.Idle;
    }

    public void IsHit()
    {
        myMovement.isHit = true;
        state = State.Idle;
    }

    public void Move()
    {
        if (Random.value > 1f && myMovement.IsGrounded())
            Dash();
        else if (Random.value > 0.4f && myMovement.IsGrounded())
            Walk();
        else if (Random.value > 0.95f && myMovement.IsGrounded())
            Jump();
        else
            state = State.Idle;
    }

    public void Walk()
    {
        if (transform.position.x > enemy.transform.position.x)
        {
            myMovement.leftPressed = true;
        }
        else
        {
            myMovement.rightPressed = true;
        }
        state = State.Idle;
    }

    public void Jump()
    {
        if (transform.position.x > enemy.transform.position.x)
            myMovement.leftPressed = true;
        else
            myMovement.rightPressed = true;
        myMovement.jumpPressed = true;
        state = State.Idle;

    }

    public void Dash()
    {
        if (transform.position.x > enemy.transform.position.x)
            myMovement.leftPressed = true;
        else
            myMovement.rightPressed = true;
        myMovement.dashPressed = true;
        state = State.Idle;

    }
}
