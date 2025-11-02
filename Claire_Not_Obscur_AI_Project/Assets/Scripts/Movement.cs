using System.Collections;
using Unity.VisualScripting;
using UnityEngine;

public class Movement : MonoBehaviour
{
    public bool isNPC = false;
    [SerializeField] BoxCollider idleHurtBox;
    [SerializeField] BoxCollider jumpingHurtBox;
    [SerializeField] BoxCollider AAHurtBox;
    [SerializeField] BoxCollider punchHitBox;
    [SerializeField] BoxCollider kickHitBox;
    [SerializeField] BoxCollider AAHitBox;

    public float jumpForce = 10f;
    public float fallMultiplier = 2.5f;
    public float lowJumpMultiplier = 2f;
    public float movementSpeed = 0.1f;
    public float horizontalAirSpeed = 0.3f;
    public float dashSpeed = 1f;
    public float hitKnockback = 300f;
    public float gravity = 9.81f;
    public float yVel = 0f;
    public float xVel = 0f;

    public KeyCode leftKey;
    public KeyCode rightKey;
    public KeyCode jumpKey;
    public KeyCode dashKey;
    public KeyCode attackKey;
    public KeyCode downKey;

    public bool leftPressed;
    public bool rightPressed;
    public bool upPressed;
    public bool jumpPressed = false;
    public bool dashPressed;
    public bool attackPressed;
    public bool downPressed;


    Rigidbody rb;
    MeshRenderer mr;

    public bool isGrounded;
    public bool isWalking;
    public bool dashing;
    public bool attacking;
    public bool isJumping;
    public bool isHit;
    public bool inHitstun = false;
    public bool locked = false;
    float movementDir;

    public Animator anim;

    [Header("Sound Effects")]
    public AudioClip attackSound;
    public AudioClip fallSound;
    private AudioSource audioSource;


    IEnumerator Attack(int startupFrames, int activeFrames, int recoveryFrames, bool state, BoxCollider activeHitbox = null, BoxCollider activeHurtbox = null, BoxCollider previousHurtbox = null)
    {
        // TODO: add startup animation
        state = true;
        if (attackSound != null && !audioSource.isPlaying){
            audioSource.PlayOneShot(attackSound);
        }

        yield return new WaitForFrames(startupFrames);

        if (activeHitbox != null)
        {
            activeHitbox.enabled = true;
        }
        if (activeHurtbox != null)
        {
            activeHurtbox.enabled = true;
        }
        if (previousHurtbox != null)
        {
            previousHurtbox.enabled = false;
        }

        yield return new WaitForFrames(activeFrames);
        if (activeHitbox != null)
        {
            activeHitbox.enabled = false;
        }


        yield return new WaitForFrames(recoveryFrames);
        if (activeHurtbox != null)
        {
            activeHurtbox.enabled = false;
        }
        if (previousHurtbox != null)
        {
            previousHurtbox.enabled = true;
        }
        state = false;
        anim.SetBool("attacking", false);
        anim.SetBool("isKicking", false);
        anim.SetBool("isAA", false);
    }

    void Start()
    {
        rb = GetComponent<Rigidbody>();
        mr = GetComponent<MeshRenderer>();

        idleHurtBox.enabled = true;
        jumpingHurtBox.enabled = false;
        AAHurtBox.enabled = false;
        punchHitBox.enabled = false;
        kickHitBox.enabled = false;
        AAHitBox.enabled = false;
        audioSource = gameObject.AddComponent<AudioSource>();
    }

    void Update()
    {
        if(!isNPC)
            UpdateInputState();

        HandleInputState();

        isGrounded = IsGrounded();

        if(transform.position.x < -8f)
        {
            rb.position = new Vector3(-8, transform.position.y, 0);
        }

        if (transform.position.x > 8f)
        {
            rb.position = new Vector3(8, transform.position.y, 0);
        }

        if (transform.position.y < -2f)
        {
            audioSource.PlayOneShot(fallSound, 0.7f);
            // instantly kill player
            if (GetComponent<Health>() != null)
            {
                GetComponent<Health>().currentHP = 0;
                GetComponent<Health>().hpBar.DecrementHP();
            }
        }
    }

    private void UpdateInputState()
    {
        leftPressed = Input.GetKey(leftKey);
        rightPressed = Input.GetKey(rightKey);
        jumpPressed = Input.GetKey(jumpKey);
        dashPressed = Input.GetKey(dashKey);
        attackPressed = Input.GetKey(attackKey);
        downPressed = Input.GetKey(downKey);
    }

    private void HandleInputState()
    {

        if (IsGrounded() && !attacking && !locked && !dashing)
        {
            if (leftPressed)
            {
                anim.SetBool("isJumping", false);
                anim.SetBool("isWalking", true);
                movementDir = -1;
            }
            else if (rightPressed)
            {
                anim.SetBool("isJumping", false);
                anim.SetBool("isWalking", true);
                movementDir = 1f;
            }
            else
            {
                anim.SetBool("isHit", false);
                anim.SetBool("isJumping", false);
                anim.SetBool("attacking", false);
                anim.SetBool("isWalking", false);
                movementDir = 0f;
            }
            if (!dashing)
            {
                anim.SetBool("isHit", false);
                anim.SetBool("isJumping", false);
                anim.SetBool("attacking", false);

                xVel = movementDir * movementSpeed;
            }
            if (downPressed && attackPressed)
            {
                anim.SetBool("isHit", false);
                anim.SetBool("isJumping", false);
                anim.SetBool("isWalking", false);
                anim.SetBool("attacking", false);
                anim.SetBool("isAA", true);
                StartCoroutine(Attack(5, 6, 10, attacking, AAHitBox, AAHurtBox, idleHurtBox));
            }
            else if (attackPressed)
            {
                anim.SetBool("isJumping", false);
                anim.SetBool("isHit", false);
                anim.SetBool("isWalking", false);
                anim.SetBool("attacking", true);
                StartCoroutine(Attack(3, 5, 6, attacking, punchHitBox));
            }
        }
        else
        {
            if (attackPressed && !attacking && !locked)
            {
                anim.SetBool("isWalking", false);
                anim.SetBool("isJumping", false);
                anim.SetBool("isHit", false);
                anim.SetBool("isKicking", true);
                StartCoroutine(Attack(3, 5, 0, attacking, kickHitBox, jumpingHurtBox));
            }
        }

        if (isHit)
        {
            anim.SetBool("isWalking", false);
            anim.SetBool("attacking", false);
            anim.SetBool("isJumping", false);
            anim.SetBool("isHit", true);
            isHit = false;
            if(movementDir == 0)
            {
                movementDir = 1f;
            }
            xVel += hitKnockback * (-1 * transform.right.x);

            StartCoroutine(Attack(0, 0, 13, locked, null, null, idleHurtBox));
        }


        if (jumpPressed && IsGrounded() && !attacking && !locked)
        {
            anim.SetBool("isHit", false);
            anim.SetBool("attacking", false);
            anim.SetBool("isWalking", false);

            anim.SetBool("isJumping", true);

            yVel = 10f;
            xVel *= horizontalAirSpeed;

        }
        if (IsGrounded())
        {
            anim.SetBool("isJumping", false);
            yVel = Mathf.Max(yVel, 0);
        }
        else
        {
            anim.SetBool("isWalking", false);
            anim.SetBool("isJumping", true);
            yVel -= gravity * Time.deltaTime;
        }

        // Dash

        if (dashing && Mathf.Abs(xVel) < 1)
        {
            anim.SetBool("isWalking", false);
            dashing = false;
            xVel = 0;
        }
        if (dashing)
        {
            anim.SetBool("isWalking", false);
            xVel -= horizontalAirSpeed * movementDir;
        }
        if (dashPressed && IsGrounded() && !dashing && !attacking && !locked)
        {
            dashing = true;
            if (movementDir == 0)
            {
                movementDir = 1f;
            }
            anim.SetBool("isWalking", false);
            xVel += dashSpeed * movementDir;
        }



        rb.position += new Vector3(xVel, yVel * Time.deltaTime, 0);
    }


    public bool IsGrounded()
    {
        return Physics.Raycast(transform.position, Vector3.down, mr.bounds.size.y/2f);
    }
}
